from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from state import CROP_SPECS, CropSpec, FarmState, Pos, Tile

MarketOrder = List[Union[str, int]]


def get_fibonacci_cost(hire_index: int, cost_mult: float = 1.0) -> float:
    """Hiring cost berdasar Fibonacci 1-indexed.

    n=1 -> 1, n=2 -> 1, n=3 -> 2, n=4 -> 3, n=5 -> 5, n=6 -> 8, n=7 -> 13, ...
    """
    if hire_index <= 0:
        return 0.0
    if hire_index in (1, 2):
        return 1.0 * cost_mult

    a, b = 1, 1
    for _ in range(3, hire_index + 1):
        a, b = b, a + b
    return float(b) * cost_mult


@dataclass(frozen=True, slots=True)
class EconomicTask:
    """Task lapangan dengan nilai moneter, biaya aksi, dan constraint waktu."""

    task_id: str
    task_type: str
    position: Pos
    economic_value: float
    slack_turns: int
    day_offset: int


@dataclass(slots=True)
class WorkerSimState:
    """State worker selama simulasi multi-hari."""

    worker_id: int
    current_pos: Pos
    daily_budgets: List[int] = field(default_factory=lambda: [24, 24, 24])


class HiringManager:
    """Labor hiring optimizer dengan Day-Budgeted Forward MDEV."""

    ACTIONS_PER_WORKER_PER_DAY: int = 24
    TOTAL_SEASON_DAYS: int = 30

    DEFAULT_PLANNING_HORIZON_DAYS: int = 5

    SHED_SPAWN_TILES: List[Pos] = [(5, 4), (4, 5), (5, 5), (4, 4)]

    # Animal product base price untuk valuasi FEED task
    ANIMAL_PRODUCT_PRICE: Dict[str, float] = {
        "GOOSE": 60.0,   # EGG
        "COW": 250.0,    # MILK
        "SHEEP": 220.0,  # WOOL
    }

    def __init__(
        self,
        farm_hand_cost_mult: float = 1.0,
        safety_margin_ratio: float = 0.0,
        # FIX: naikkan buffer ke 400 — hiring terlalu agresif di 150
        min_cash_buffer: float = 400.0,
        # FIX: turunkan max_hands ke 6 — 9-10 hands = labor $495/hari tidak worth
        max_hands: int = 6,
        planning_horizon_days: int = DEFAULT_PLANNING_HORIZON_DAYS,
        end_of_season_decay_start: int = 25,
    ) -> None:
        self.farm_hand_cost_mult = farm_hand_cost_mult
        self.safety_margin_ratio = safety_margin_ratio
        self.min_cash_buffer = min_cash_buffer
        self.max_hands = max_hands
        self.PLANNING_HORIZON_DAYS = max(1, planning_horizon_days)
        self.end_of_season_decay_start = end_of_season_decay_start

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _manhattan_distance(pos1: Pos, pos2: Pos) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def _end_of_season_factor(current_day: int, total_days: int, decay_start: int) -> float:
        if current_day < decay_start:
            return 1.0
        remaining = max(0, total_days - current_day)
        decay_window = max(1, total_days - decay_start)
        return 0.25 + 0.75 * (remaining / decay_window)

    # ------------------------------------------------------------------
    # Valuasi per-task
    # ------------------------------------------------------------------
    @staticmethod
    def _water_value_per_day(spec: Optional[CropSpec], price: float) -> float:
        if spec is None:
            return 1.0 * price
        growth_days = max(1, spec.max_yield_day - spec.first_yield_day)
        yield_gain_per_day = max(0.0, float(spec.max_yield - spec.base_yield)) / growth_days
        yield_gain_per_day = max(1.0, yield_gain_per_day)
        return yield_gain_per_day * price

    @staticmethod
    def _projected_yield(
        spec: Optional[CropSpec],
        simulated_age: int,
        current_yield_units: int,
    ) -> float:
        if spec is None:
            return float(max(current_yield_units, 2))
        if simulated_age < spec.first_yield_day:
            return float(max(current_yield_units, spec.base_yield))
        if spec.max_yield_day > spec.first_yield_day:
            days_in_window = max(1, spec.max_yield_day - spec.first_yield_day)
            days_elapsed = max(0, simulated_age - spec.first_yield_day)
            growth_per_day = (spec.max_yield - spec.base_yield) / float(days_in_window)
            projected = spec.base_yield + growth_per_day * days_elapsed
            projected = min(float(projected), float(spec.max_yield))
        else:
            projected = float(spec.max_yield)
        return max(projected, float(current_yield_units))

    # ------------------------------------------------------------------
    # Task extraction (FIX: tambah animal tasks)
    # ------------------------------------------------------------------
    def _extract_horizon_tasks(self, state: FarmState) -> List[EconomicTask]:
        tasks: List[EconomicTask] = []
        unlocked_positions = state.get_unlocked_tiles()
        tiles = state.tiles

        eos_factor = self._end_of_season_factor(
            state.day, self.TOTAL_SEASON_DAYS, self.end_of_season_decay_start
        )

        # Hitung kebutuhan animal
        remaining_days_total = max(1, self.TOTAL_SEASON_DAYS - state.day)
        shed_goose = state.shed.get("GOOSE", 0)
        shed_cow = state.shed.get("COW", 0)
        shed_sheep = state.shed.get("SHEEP", 0)

        # Cek ada coop/pasture untuk place
        empty_coop_exists = any(
            isinstance(t, dict) and t.get("kind") == "COOP" and not t.get("animal")
            for row in tiles for t in row
        )
        empty_pasture_exists = any(
            isinstance(t, dict) and t.get("kind") == "PASTURE" and not t.get("animal")
            for row in tiles for t in row
        )
        any_coop_exists = any(
            isinstance(t, dict) and t.get("kind") == "COOP"
            for row in tiles for t in row
        )
        any_pasture_exists = any(
            isinstance(t, dict) and t.get("kind") == "PASTURE"
            for row in tiles for t in row
        )

        for pos in unlocked_positions:
            tx, ty = pos
            tile: Tile = tiles[ty][tx] if ty < len(tiles) and tx < len(tiles[ty]) else None
            tile_key = f"{tx}_{ty}"

            # ============================================================
            # EMPTY TILE
            # ============================================================
            if tile is None:
                # ---------- PLANT ----------
                best_seed_val = 0.0
                for seed_type, count in state.seeds.items():
                    if count > 0 and seed_type in CROP_SPECS:
                        spec = CROP_SPECS[seed_type]
                        price = state.market_prices.get(spec.product_name, 15.0)
                        gross_return = price * float(spec.max_yield)
                        net_lifecycle_profit = gross_return - spec.seed_cost
                        daily_amortized_value = net_lifecycle_profit / max(
                            1.0, float(spec.first_yield_day)
                        )
                        if daily_amortized_value > best_seed_val:
                            best_seed_val = daily_amortized_value

                if best_seed_val > 0.0:
                    tasks.append(
                        EconomicTask(
                            task_id=f"plant_{tile_key}",
                            task_type="PLANT",
                            position=pos,
                            economic_value=best_seed_val * eos_factor,
                            slack_turns=48,
                            day_offset=0,
                        )
                    )

                # ---------- BUILD COOP (FIX) ----------
                if shed_goose > 0 and not any_coop_exists:
                    tasks.append(
                        EconomicTask(
                            task_id=f"build_coop_{tile_key}",
                            task_type="BUILD",
                            position=pos,
                            economic_value=remaining_days_total * 50.0 * eos_factor,
                            slack_turns=24,
                            day_offset=0,
                        )
                    )

                # ---------- BUILD PASTURE (FIX) ----------
                if (shed_cow > 0 or shed_sheep > 0) and not any_pasture_exists:
                    price = self.ANIMAL_PRODUCT_PRICE["COW"]
                    tasks.append(
                        EconomicTask(
                            task_id=f"build_pasture_{tile_key}",
                            task_type="BUILD",
                            position=pos,
                            economic_value=remaining_days_total * price * 0.5 * eos_factor,
                            slack_turns=24,
                            day_offset=0,
                        )
                    )

            elif isinstance(tile, dict):
                kind = tile.get("kind")

                # ============================================================
                # WEED
                # ============================================================
                if kind == "WEED":
                    tasks.append(
                        EconomicTask(
                            task_id=f"dig_{tile_key}",
                            task_type="DIG",
                            position=pos,
                            economic_value=15.0 * eos_factor,
                            slack_turns=24,
                            day_offset=0,
                        )
                    )

                # ============================================================
                # PLANT (crop)
                # ============================================================
                elif kind == "PLANT":
                    crop = str(tile.get("crop", ""))
                    planted_day = int(tile.get("planted_day", 0))
                    yield_units = int(tile.get("yield_units", 0))
                    crop_age = state.day - planted_day
                    spec = CROP_SPECS.get(crop)

                    product_key = spec.product_name if spec else crop
                    price = state.market_prices.get(product_key, 20.0)
                    first_yield = spec.first_yield_day if spec else 2
                    max_yield_day = spec.max_yield_day if spec else 4
                    is_one_time = (spec.crop_type == "one_time") if spec else True

                    water_value_per_day = self._water_value_per_day(spec, price) * eos_factor
                    harvest_scheduled = False

                    for day_h in range(self.PLANNING_HORIZON_DAYS):
                        simulated_age = crop_age + day_h

                        # 1. HARVEST
                        if simulated_age >= first_yield and not harvest_scheduled:
                            projected_yield = self._projected_yield(
                                spec, simulated_age, yield_units
                            )
                            harvest_val = float(projected_yield * price) * eos_factor
                            remaining_days = max(1, max_yield_day - simulated_age)
                            slack_turns = max(1, remaining_days * 24)
                            tasks.append(
                                EconomicTask(
                                    task_id=f"harvest_{tile_key}_d{day_h}",
                                    task_type="HARVEST",
                                    position=pos,
                                    economic_value=harvest_val,
                                    slack_turns=slack_turns,
                                    day_offset=day_h,
                                )
                            )
                            harvest_scheduled = True
                            if not is_one_time:
                                harvest_scheduled = False

                        # 2. WATER
                        elif day_h == 0 and not tile.get("watered_today", False):
                            slack_turns = max(1, 24 - state.hour)
                            tasks.append(
                                EconomicTask(
                                    task_id=f"water_{tile_key}_d0",
                                    task_type="WATER",
                                    position=pos,
                                    economic_value=water_value_per_day,
                                    slack_turns=slack_turns,
                                    day_offset=0,
                                )
                            )
                        elif day_h > 0 and simulated_age < first_yield:
                            tasks.append(
                                EconomicTask(
                                    task_id=f"water_{tile_key}_d{day_h}",
                                    task_type="WATER",
                                    position=pos,
                                    economic_value=water_value_per_day,
                                    slack_turns=24,
                                    day_offset=day_h,
                                )
                            )

                # ============================================================
                # COOP / PASTURE (FIX: tambah FEED, PLACE, HARVEST, COLLECT)
                # ============================================================
                elif kind in ("COOP", "PASTURE"):
                    animal = tile.get("animal")
                    if animal:
                        animal_key = str(animal).upper()
                        product_price = self.ANIMAL_PRODUCT_PRICE.get(
                            animal_key, 50.0
                        )

                        # ---------- FEED (kritis) ----------
                        if not tile.get("fed_today", False):
                            # Kalau tidak di-feed 2 hari, animal mati
                            # Nilai = seluruh future income
                            feed_value = remaining_days_total * product_price * eos_factor
                            tasks.append(
                                EconomicTask(
                                    task_id=f"feed_{tile_key}",
                                    task_type="FEED",
                                    position=pos,
                                    economic_value=feed_value,
                                    slack_turns=24,
                                    day_offset=0,
                                )
                            )

                        # ---------- HARVEST egg/milk/wool ----------
                        if int(tile.get("yield_units", 0)) > 0:
                            yield_val = int(tile.get("yield_units", 0)) * product_price * eos_factor
                            tasks.append(
                                EconomicTask(
                                    task_id=f"harvest_animal_{tile_key}",
                                    task_type="HARVEST",
                                    position=pos,
                                    economic_value=yield_val,
                                    slack_turns=48,
                                    day_offset=0,
                                )
                            )

                        # ---------- COLLECT FERTILIZER ----------
                        if tile.get("fertilizer_available", False):
                            tasks.append(
                                EconomicTask(
                                    task_id=f"collect_fert_{tile_key}",
                                    task_type="COLLECT_FERTILIZER",
                                    position=pos,
                                    economic_value=100.0 * eos_factor,
                                    slack_turns=48,
                                    day_offset=0,
                                )
                            )

                    else:
                        # ---------- PLACE animal dari shed ----------
                        if kind == "COOP" and shed_goose > 0:
                            tasks.append(
                                EconomicTask(
                                    task_id=f"place_goose_{tile_key}",
                                    task_type="PLACE",
                                    position=pos,
                                    economic_value=remaining_days_total * 60.0 * eos_factor,
                                    slack_turns=48,
                                    day_offset=0,
                                )
                            )
                        if kind == "PASTURE" and (shed_cow > 0 or shed_sheep > 0):
                            price = self.ANIMAL_PRODUCT_PRICE["COW"]
                            tasks.append(
                                EconomicTask(
                                    task_id=f"place_cow_{tile_key}",
                                    task_type="PLACE",
                                    position=pos,
                                    economic_value=remaining_days_total * price * 0.5 * eos_factor,
                                    slack_turns=48,
                                    day_offset=0,
                                )
                            )

        return tasks

    # ------------------------------------------------------------------
    # Worker simulation
    # ------------------------------------------------------------------
    def _get_initial_worker_states(
        self, state: FarmState, num_workers: int
    ) -> List[WorkerSimState]:
        workers: List[WorkerSimState] = []

        workers.append(
            WorkerSimState(
                worker_id=0,
                current_pos=state.farmer_pos,
                daily_budgets=[self.ACTIONS_PER_WORKER_PER_DAY] * self.PLANNING_HORIZON_DAYS,
            )
        )

        for idx, hand_pos in enumerate(state.hands_pos):
            if len(workers) < num_workers:
                workers.append(
                    WorkerSimState(
                        worker_id=idx + 1,
                        current_pos=hand_pos,
                        daily_budgets=[self.ACTIONS_PER_WORKER_PER_DAY] * self.PLANNING_HORIZON_DAYS,
                    )
                )

        spawn_idx = 0
        while len(workers) < num_workers:
            spawn_pos = self.SHED_SPAWN_TILES[spawn_idx % len(self.SHED_SPAWN_TILES)]
            workers.append(
                WorkerSimState(
                    worker_id=len(workers),
                    current_pos=spawn_pos,
                    daily_budgets=[self.ACTIONS_PER_WORKER_PER_DAY] * self.PLANNING_HORIZON_DAYS,
                )
            )
            spawn_idx += 1

        return workers

    def calculate_forward_expected_value(
        self,
        num_workers: int,
        state: FarmState,
        tasks: Sequence[EconomicTask],
    ) -> float:
        """EV_{t:t+H}(N) via greedy discrete assignment per hari."""
        if num_workers <= 0 or not tasks:
            return 0.0

        workers = self._get_initial_worker_states(state, num_workers)
        spawn_positions: List[Pos] = [w.current_pos for w in workers]

        completed_task_ids: Set[str] = set()
        total_realized_value = 0.0

        for day_h in range(self.PLANNING_HORIZON_DAYS):
            for idx, w in enumerate(workers):
                w.current_pos = spawn_positions[idx]

            day_tasks = [
                t for t in tasks
                if t.day_offset == day_h and t.task_id not in completed_task_ids
            ]
            if not day_tasks:
                continue

            while True:
                best_assignment: Optional[Tuple[int, EconomicTask, int, float]] = None
                max_priority = -1.0

                for w_idx, worker in enumerate(workers):
                    budget_remaining = worker.daily_budgets[day_h]
                    if budget_remaining <= 0:
                        continue

                    for task in day_tasks:
                        if task.task_id in completed_task_ids:
                            continue

                        action_cost = (
                            self._manhattan_distance(worker.current_pos, task.position) + 1
                        )
                        if action_cost > budget_remaining:
                            continue

                        discount = 1.0 / (1.0 + 0.15 * float(day_h))
                        urgency_mult = 1.0 + (1.0 / float(task.slack_turns + 1))
                        priority = (
                            task.economic_value / float(action_cost)
                        ) * urgency_mult * discount

                        if priority > max_priority:
                            max_priority = priority
                            best_assignment = (w_idx, task, action_cost, task.economic_value)

                if best_assignment is None:
                    break

                assigned_w_idx, assigned_task, cost_spent, task_val = best_assignment
                workers[assigned_w_idx].daily_budgets[day_h] -= cost_spent
                workers[assigned_w_idx].current_pos = assigned_task.position
                completed_task_ids.add(assigned_task.task_id)
                total_realized_value += task_val

        return total_realized_value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan_hiring_orders(self, state: FarmState) -> List[MarketOrder]:
        """Hire jika ΔEV_{t:t+H}(N) > hire_cost, dengan guard cash buffer & max hands."""
        orders: List[MarketOrder] = []

        if state.day >= (self.TOTAL_SEASON_DAYS - 2):
            return orders

        if state.hour != 0:
            return orders

        current_hires = state.hires_today
        current_workers = 1 + current_hires
        if current_workers >= self.max_hands:
            return orders

        field_tasks = self._extract_horizon_tasks(state)
        if not field_tasks:
            return orders

        ev_current = self.calculate_forward_expected_value(
            current_workers, state, field_tasks
        )

        simulated_hires = current_hires
        simulated_workers = current_workers
        ev_previous = ev_current

        while True:
            next_hire_index = simulated_hires + 1
            next_worker_count = simulated_workers + 1

            if next_worker_count > self.max_hands:
                break

            hire_cost = get_fibonacci_cost(next_hire_index, self.farm_hand_cost_mult)

            cash_after_hire = state.money - hire_cost
            if cash_after_hire < self.min_cash_buffer:
                break

            ev_next = self.calculate_forward_expected_value(
                next_worker_count, state, field_tasks
            )

            marginal_economic_value = ev_next - ev_previous
            required_threshold = hire_cost * (1.0 + self.safety_margin_ratio)

            if marginal_economic_value > required_threshold:
                orders.append(["HIRE"])
                simulated_hires += 1
                simulated_workers += 1
                ev_previous = ev_next
            else:
                break

        return orders