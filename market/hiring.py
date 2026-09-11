"""Forward-Looking Marginal Daily Economic Value (MDEV) Labor Optimizer for Kaggriculture.

Provides day-budgeted, discrete task-worker assignment modeling over a multi-day horizon,
preventing duplicate task evaluations, enforcing temporal execution constraints, and
accurately accounting for crop lifecycle values against the Fibonacci hiring cost curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from state import CROP_SPECS, CropSpec, FarmState, Pos, Tile

MarketOrder = List[Union[str, int]]


def get_fibonacci_cost(hire_index: int, cost_mult: float = 1.0) -> float:
    """Calculates hiring cost based on 1-indexed Fibonacci sequence.

    n=1 -> 1.0, n=2 -> 1.0, n=3 -> 2.0, n=4 -> 3.0, n=5 -> 5.0, ...
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
    """Encapsulates a field task with monetary value, action cost, and temporal constraints."""

    task_id: str
    task_type: str
    position: Pos
    economic_value: float  # Pure monetary value or loss avoidance ($)
    slack_turns: int       # Turns remaining before deadline/decay
    day_offset: int        # Day within planning horizon H (0..H-1)


@dataclass(slots=True)
class WorkerSimState:
    """Tracks state and remaining action budget for a worker during multi-day simulation."""

    worker_id: int
    current_pos: Pos
    daily_budgets: List[int] = field(default_factory=lambda: [24, 24, 24])


class HiringManager:
    """Labor hiring optimizer utilizing Day-Budgeted Forward MDEV over horizon H=3."""

    ACTIONS_PER_WORKER_PER_DAY: int = 24
    TOTAL_SEASON_DAYS: int = 30
    PLANNING_HORIZON_DAYS: int = 3  # Forward lookahead window (H)

    # Official Shed-adjacent spawn tile positions for newly hired farm hands
    SHED_SPAWN_TILES: List[Pos] = [(5, 4), (4, 5), (5, 5), (4, 4)]

    def __init__(
        self,
        farm_hand_cost_mult: float = 1.0,
        safety_margin_ratio: float = 0.0,
    ) -> None:
        self.farm_hand_cost_mult = farm_hand_cost_mult
        self.safety_margin_ratio = safety_margin_ratio

    @staticmethod
    def _manhattan_distance(pos1: Pos, pos2: Pos) -> int:
        """Calculates Manhattan distance between two tile coordinates."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _extract_horizon_tasks(self, state: FarmState) -> List[EconomicTask]:
        """Extracts discrete, deduplicated tasks across the current day and planning horizon H."""
        tasks: List[EconomicTask] = []
        unlocked_positions = state.get_unlocked_tiles()
        tiles = state.tiles

        for pos in unlocked_positions:
            tx, ty = pos
            tile: Tile = tiles[ty][tx] if ty < len(tiles) and tx < len(tiles[ty]) else None
            tile_key = f"{tx}_{ty}"

            if tile is None:
                # Task: PLANT (Net Lifecycle Return amortized across growth cycle)
                best_seed_val = 0.0
                for seed_type, count in state.seeds.items():
                    if count > 0 and seed_type in CROP_SPECS:
                        spec = CROP_SPECS[seed_type]
                        price = state.market_prices.get(spec.product_name, 15.0)

                        gross_return = price * float(spec.max_yield)
                        net_lifecycle_profit = gross_return - spec.seed_cost
                        daily_amortized_value = net_lifecycle_profit / max(1.0, float(spec.first_yield_day))

                        if daily_amortized_value > best_seed_val:
                            best_seed_val = daily_amortized_value

                if best_seed_val > 0.0:
                    tasks.append(
                        EconomicTask(
                            task_id=f"plant_{tile_key}",
                            task_type="PLANT",
                            position=pos,
                            economic_value=best_seed_val,
                            slack_turns=48,
                            day_offset=0,  # Planting is scheduled on day 0
                        )
                    )

            elif isinstance(tile, dict):
                kind = tile.get("kind")

                if kind == "WEED":
                    # Task: DIG WEED
                    tasks.append(
                        EconomicTask(
                            task_id=f"dig_{tile_key}",
                            task_type="DIG",
                            position=pos,
                            economic_value=15.0,
                            slack_turns=24,
                            day_offset=0,
                        )
                    )

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
                    is_one_time = spec.crop_type == "one_time" if spec else True

                    harvest_scheduled = False

                    # Project over multi-day horizon H
                    for day_h in range(self.PLANNING_HORIZON_DAYS):
                        simulated_age = crop_age + day_h

                        # 1. HARVEST Projection (Deduplicated for one-time crops)
                        if simulated_age >= first_yield:
                            if is_one_time and harvest_scheduled:
                                # Prevent multi-counting single harvest event
                                continue

                            harvest_val = float(max(yield_units, spec.base_yield if spec else 2) * price)
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

                        # 2. WATER Projection (Assigned strictly to its specific day_offset)
                        elif day_h == 0 and not tile.get("watered_today", False):
                            base_y = float(spec.base_yield) if spec else 1.0
                            loss_avoidance = base_y * price
                            slack_turns = max(1, 24 - state.hour)

                            tasks.append(
                                EconomicTask(
                                    task_id=f"water_{tile_key}_d0",
                                    task_type="WATER",
                                    position=pos,
                                    economic_value=loss_avoidance,
                                    slack_turns=slack_turns,
                                    day_offset=0,
                                )
                            )
                        elif day_h > 0 and simulated_age < first_yield:
                            # Mandatory future day watering
                            base_y = float(spec.base_yield) if spec else 1.0
                            loss_avoidance = base_y * price

                            tasks.append(
                                EconomicTask(
                                    task_id=f"water_{tile_key}_d{day_h}",
                                    task_type="WATER",
                                    position=pos,
                                    economic_value=loss_avoidance,
                                    slack_turns=24,
                                    day_offset=day_h,
                                )
                            )

        return tasks

    def _get_initial_worker_states(self, state: FarmState, num_workers: int) -> List[WorkerSimState]:
        """Builds simulation states for active workers and simulated extra hands with correct spawn positions."""
        workers: List[WorkerSimState] = []

        # 1. Main Farmer
        workers.append(
            WorkerSimState(
                worker_id=0,
                current_pos=state.farmer_pos,
                daily_budgets=[self.ACTIONS_PER_WORKER_PER_DAY] * self.PLANNING_HORIZON_DAYS,
            )
        )

        # 2. Active Hired Hands
        for idx, hand_pos in enumerate(state.hands_pos):
            if len(workers) < num_workers:
                workers.append(
                    WorkerSimState(
                        worker_id=idx + 1,
                        current_pos=hand_pos,
                        daily_budgets=[self.ACTIONS_PER_WORKER_PER_DAY] * self.PLANNING_HORIZON_DAYS,
                    )
                )

        # 3. Simulated extra workers starting at official Shed spawn tiles
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
        self, num_workers: int, state: FarmState, tasks: Sequence[EconomicTask]
    ) -> float:
        """Computes EV_{t:t+H}(N) using day-budgeted discrete assignment across horizon H."""
        if num_workers <= 0 or not tasks:
            return 0.0

        workers = self._get_initial_worker_states(state, num_workers)
        completed_task_ids: Set[str] = set()
        total_realized_value = 0.0

        # Simulate day-by-day execution to enforce strictly 24 actions per worker per day
        for day_h in range(self.PLANNING_HORIZON_DAYS):
            # Filter tasks scheduled specifically for this day
            day_tasks = [t for t in tasks if t.day_offset == day_h and t.task_id not in completed_task_ids]
            if not day_tasks:
                continue

            # Greedy discrete bipartite matching for current day
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

                        # Individual Action Cost: Manhattan distance + 1 execution turn
                        action_cost = self._manhattan_distance(worker.current_pos, task.position) + 1

                        if action_cost <= budget_remaining:
                            # Discount factor for future day offsets
                            discount = 1.0 / (1.0 + (0.15 * float(day_h)))
                            urgency_mult = 1.0 + (1.0 / float(task.slack_turns + 1))
                            priority = (task.economic_value / float(action_cost)) * urgency_mult * discount

                            if priority > max_priority:
                                max_priority = priority
                                best_assignment = (w_idx, task, action_cost, task.economic_value)

                if best_assignment is None:
                    break

                assigned_w_idx, assigned_task, cost_spent, task_val = best_assignment

                # Commit assignment to worker state for current day
                workers[assigned_w_idx].daily_budgets[day_h] -= cost_spent
                workers[assigned_w_idx].current_pos = assigned_task.position
                completed_task_ids.add(assigned_task.task_id)
                total_realized_value += task_val

        return total_realized_value

    def plan_hiring_orders(self, state: FarmState) -> List[MarketOrder]:
        """Evaluates Forward MDEV_n > H_n and dispatches optimal hiring orders."""
        orders: List[MarketOrder] = []

        if state.day >= (self.TOTAL_SEASON_DAYS - 2):
            return orders

        if state.hour != 0:
            return orders

        field_tasks = self._extract_horizon_tasks(state)
        if not field_tasks:
            return orders

        current_hires = state.hires_today
        current_workers = 1 + current_hires

        # Calculate baseline Forward EV_{t:t+H}(N)
        ev_current = self.calculate_forward_expected_value(current_workers, state, field_tasks)

        simulated_hires = current_hires
        simulated_workers = current_workers
        ev_previous = ev_current

        while True:
            next_hire_index = simulated_hires + 1
            next_worker_count = simulated_workers + 1

            hire_cost = get_fibonacci_cost(next_hire_index, self.farm_hand_cost_mult)
            ev_next = self.calculate_forward_expected_value(next_worker_count, state, field_tasks)

            # Forward MDEV_n = EV_{t:t+H}(N+1) - EV_{t:t+H}(N)
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