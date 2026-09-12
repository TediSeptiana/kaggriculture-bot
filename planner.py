"""High-level orchestration agent applying market demand-driven worker dispatching and global target persistence."""

from __future__ import annotations

from typing import Any, Dict, List, Set
from market import MarketPlanner
from state import CROP_SPECS, FarmState
from worker import Pos, WorkerPlanner, WorkerRole


class AgentPlanner:
    """Orchestrates state parsing, global target persistence, and demand-driven worker dispatching."""

    def __init__(self) -> None:
        self.worker_planner = WorkerPlanner()
        self.market_planner = MarketPlanner()

    def _assess_global_farm_demand(self, state: FarmState) -> List[WorkerRole]:
        """Evaluates macro demand across the farm to create a dynamic role priority list."""
        unlocked = state.get_unlocked_tiles()
        tiles = state.tiles

        unwatered_count = 0
        harvestable_count = 0
        weed_count = 0
        empty_count = 0

        for pos in unlocked:
            tx, ty = pos
            if ty >= len(tiles) or tx >= len(tiles[ty]):
                continue
            t = tiles[ty][tx]

            if t is None:
                empty_count += 1
            elif isinstance(t, dict):
                kind = t.get("kind")
                if kind == "WEED":
                    weed_count += 1
                elif kind == "PLANT":
                    if not t.get("watered_today", False):
                        unwatered_count += 1

                    crop = str(t.get("crop", ""))
                    planted_day = int(t.get("planted_day", 0))
                    yield_units = int(t.get("yield_units", 0))
                    crop_age = state.day - planted_day
                    spec = CROP_SPECS.get(crop)
                    first_yield = spec.first_yield_day if spec else 2

                    if crop_age >= first_yield and yield_units > 0:
                        harvestable_count += 1

        # Build demand priority order based on macro urgency
        role_demands: List[WorkerRole] = []

        # Urgent Priority 1: Harvesting mature crops (Direct Cash Liquidation)
        if harvestable_count > 0:
            role_demands.extend([WorkerRole.HARVESTER] * min(3, harvestable_count))

        # Urgent Priority 2: Watering thirsty crops (Prevent Yield Degradation)
        if unwatered_count > 0:
            role_demands.extend([WorkerRole.WATERER] * min(4, unwatered_count))

        # Priority 3: Planting if seeds are available and empty tiles exist
        total_seeds = sum(state.seeds.values())
        if empty_count > 0 and total_seeds > 0:
            role_demands.extend([WorkerRole.PLANTER] * min(2, empty_count))

        # Priority 4: Clearing weeds
        if weed_count > 0:
            role_demands.extend([WorkerRole.DIGGER] * min(2, weed_count))

        # Fallback to VERSATILE if no specific dominant demand
        if not role_demands:
            role_demands = [WorkerRole.VERSATILE]

        return role_demands

    def plan_turn(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point processing observation dict and returning action payload."""
        state = FarmState.from_obs(obs)

        # 1. Market Queue Planning (Orders executed sequentially)
        market_actions = self.market_planner.plan_orders(state)

        # 2. Dynamic Macro Demand Assessment
        demand_roles = self._assess_global_farm_demand(state)

        # Prevent multiple units from selecting the same tile during this turn.
        assigned_targets: Set[Pos] = set()

        # 3. Main Farmer Execution (Unit ID: 0)
        farmer_inv = state.inventories[0] if len(state.inventories) > 0 else []
        farmer_role = demand_roles[0] if demand_roles else WorkerRole.DIGGER

        farmer_action = self.worker_planner.decide_action(
            unit_pos=state.farmer_pos,
            state=state,
            inventory=farmer_inv,
            role=farmer_role,
            assigned_targets=assigned_targets,
        )

        # 4. Hired Hands Execution (Unit IDs: 1..N)
        hands_actions: List[List[Any]] = []
        for idx, hand_pos in enumerate(state.hands_pos):
            hand_unit_id = idx + 1
            hand_inv = (
                state.inventories[hand_unit_id]
                if hand_unit_id < len(state.inventories)
                else []
            )

            # Assign role dynamically based on macro farm demand
            role_idx = (idx + 1) % len(demand_roles)
            assigned_role = demand_roles[role_idx]

            hand_act = self.worker_planner.decide_action(
                unit_pos=hand_pos,
                state=state,
                inventory=hand_inv,
                role=assigned_role,
                assigned_targets=assigned_targets,
            )
            hands_actions.append(hand_act)

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": market_actions,
        }