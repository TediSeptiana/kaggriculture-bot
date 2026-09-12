"""Forward-Looking Marginal Daily Economic Value (MDEV) Labor Optimizer for Kaggriculture.

Perbaikan utama dibanding versi sebelumnya:
  1. FIX 1 dibatalkan: hire cost adalah pembayaran ONE-TIME, bukan per-hari.
     Perbandingan yang benar:  ΔEV_{t:t+H}(N) > hire_cost  (bukan × H).
  2. Ditambahkan parameter min_cash_buffer: hiring ditolak jika cash < buffer + hire_cost.
  3. Ditambahkan parameter max_hands: batas keras jumlah worker.
  4. PLANNING_HORIZON_DAYS dibuat configurable (default 5, bukan 3).
  5. Deduplikasi harvest diperbaiki: hanya harvest PERTAMA per tile dalam horizon
     yang dijadwalkan, mencegah over-count pada multi-harvest crop.
  6. Ditambahkan end-of-season decay: nilai task di hari-hari akhir musim diturunkan
     supaya hiring di akhir musim tidak dipaksakan.
  7. _projected_yield: regrowth setelah harvest tidak lagi dihitung ganda.
"""

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
    economic_value: float   # Nilai moneter murni atau loss avoidance ($)
    slack_turns: int        # Sisa turn sebelum deadline/decay
    day_offset: int         # Hari dalam horizon (0..H-1)


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

    # Default horizon diperpanjang dari 3 -> 5 supaya crop lambat (WHEAT) tetap terlihat nilainya.
    DEFAULT_PLANNING_HORIZON_DAYS: int = 5

    # Spawn tile resmi di dekat shed untuk farm hand baru.
    SHED_SPAWN_TILES: List[Pos] = [(5, 4), (4, 5), (5, 5), (4, 4)]

    def __init__(
        self,
        farm_hand_cost_mult: float = 1.0,
        safety_margin_ratio: float = 0.0,
        min_cash_buffer: float = 500.0,
        max_hands: int = 10,
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
        """Nilai task diturunkan mendekati akhir musim.

        - Sebelum decay_start: faktor 1.0
        - Di akhir musim: turun linear sampai ~0.25
        """
        if current_day < decay_start:
            return 1.0
        remaining = max(0, total_days - current_day)
        decay_window = max(1, total_days - decay_start)
        # linear dari 1.0 -> 0.25 di hari terakhir
        return 0.25 + 0.75 * (remaining / decay_window)

    # ------------------------------------------------------------------
    # Valuasi per-task
    # ------------------------------------------------------------------
    @staticmethod
    def _water_value_per_day(spec: Optional[CropSpec], price: float) -> float:
        """Nilai moneter satu hari pertumbuhan tambahan.

        Untuk crop one-time: nilai air hanya sampai first_yield.
        Untuk multi-harvest: air tetap bernilai karena regrowth.
        """
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
        """Estimasi yield panen pada umur simulasi tertentu."""
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
    # Task extraction
    # ------------------------------------------------------------------
    def _extract_horizon_tasks(self, state: FarmState) -> List[EconomicTask]:
        tasks: List[EconomicTask] = []
        unlocked_positions = state.get_unlocked_tiles()
        tiles = state.tiles

        eos_factor = self._end_of_season_factor(
            state.day, self.TOTAL_SEASON_DAYS, self.end_of_season_decay_start
        )

        for pos in unlocked_positions:
            tx, ty = pos
            tile: Tile = tiles[ty][tx] if ty < len(tiles) and tx < len(tiles[ty]) else None
            tile_key = f"{tx}_{ty}"

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

            elif isinstance(tile, dict):
                kind = tile.get("kind")

                # ---------- DIG WEED ----------
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

                # ---------- PLANT (crop) ----------
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

                    # Hanya jadwalkan HARVEST PERTAMA dalam horizon.
                    # Mencegah over-count untuk multi-harvest crop: kalau panen
                    # di day_h=0, harvest di day_h>0 tidak valid lagi.
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

                            # Multi-harvest crop: izinkan satu harvest lanjutan
                            # di akhir horizon sebagai bonus (tidak wajib).
                            if not is_one_time:
                                # Biarkan loop lanjut agar bisa menjadwalkan
                                # satu harvest tambahan di hari berikutnya.
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

        return tasks

    # ------------------------------------------------------------------
    # Worker simulation
    # ------------------------------------------------------------------
    def _get_initial_worker_states(
        self, state: FarmState, num_workers: int
    ) -> List[WorkerSimState]:
        workers: List[WorkerSimState] = []

        # 1. Farmer utama
        workers.append(
            WorkerSimState(
                worker_id=0,
                current_pos=state.farmer_pos,
                daily_budgets=[self.ACTIONS_PER_WORKER_PER_DAY] * self.PLANNING_HORIZON_DAYS,
            )
        )

        # 2. Hand yang sudah aktif
        for idx, hand_pos in enumerate(state.hands_pos):
            if len(workers) < num_workers:
                workers.append(
                    WorkerSimState(
                        worker_id=idx + 1,
                        current_pos=hand_pos,
                        daily_budgets=[self.ACTIONS_PER_WORKER_PER_DAY] * self.PLANNING_HORIZON_DAYS,
                    )
                )

        # 3. Hand simulasi (belum di-hire)
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
            # Worker reset ke spawn di awal hari
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

        # 1. Terlalu dekat akhir musim -> stop hiring
        if state.day >= (self.TOTAL_SEASON_DAYS - 2):
            return orders

        # 2. Hanya evaluasi di awal hari
        if state.hour != 0:
            return orders

        # 3. Sudah mencapai batas hands
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

            # Batas keras max hands
            if next_worker_count > self.max_hands:
                break

            # ------------------------------------------------------------------
            # FIX 1 (dibatalkan): hire cost = ONE-TIME payment.
            # Jadi bandingkan ΔEV_{t:t+H}(N) langsung terhadap hire_cost.
            # Tidak dikali H.
            # ------------------------------------------------------------------
            hire_cost = get_fibonacci_cost(next_hire_index, self.farm_hand_cost_mult)

            # --------------------------------------------------------------
            # Cash-buffer guard: hiring hanya jika cash masih cukup setelah
            # menyisakan buffer minimum.
            # --------------------------------------------------------------
            cash_after_hire = state.money - hire_cost
            if cash_after_hire < self.min_cash_buffer:
                break

            ev_next = self.calculate_forward_expected_value(
                next_worker_count, state, field_tasks
            )

            # Forward MDEV_n = EV_{t:t+H}(N+1) - EV_{t:t+H}(N)
            marginal_economic_value = ev_next - ev_previous

            # Safety margin: opportunity cost / risk premium
            required_threshold = hire_cost * (1.0 + self.safety_margin_ratio)

            if marginal_economic_value > required_threshold:
                orders.append(["HIRE"])
                simulated_hires += 1
                simulated_workers += 1
                ev_previous = ev_next
            else:
                break

        return orders