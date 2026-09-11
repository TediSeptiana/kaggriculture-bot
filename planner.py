"""High-level orchestration agent applying specialized worker role dispatching."""

from __future__ import annotations

from typing import Any, Dict, List, Set
from market import MarketPlanner
from state import FarmState
from worker import Pos, WorkerPlanner, WorkerRole


class AgentPlanner:
    """Orchestrates state parsing and dispatches specialized roles to workers."""

    def __init__(self) -> None:
        self.worker_planner = WorkerPlanner()
        self.market_planner = MarketPlanner()

    def plan_turn(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point processing observation dict and returning action payload."""
        state = FarmState.from_obs(obs)

        # 1. Market Queue Planning
        market_actions = self.market_planner.plan_orders(state)

        # Global target tracker untuk mencegah 2 unit menuju tile yang sama
        assigned_targets: Set[Pos] = set()

        # 2. Main Farmer Planning (Role: DIGGER)
        farmer_inv = state.inventories[0] if len(state.inventories) > 0 else []
        farmer_action = self.worker_planner.decide_action(
            unit_pos=state.farmer_pos,
            state=state,
            inventory=farmer_inv,
            role=WorkerRole.DIGGER,
            assigned_targets=assigned_targets,
        )

        # 3. Hired Hands Planning (Specialized Roles Index Mapping)
        # Mapping: Hand 0 -> PLANTER, Hand 1 -> WATERER, Hand 2 -> HARVESTER
        hand_roles = [
            WorkerRole.PLANTER,
            WorkerRole.WATERER,
            WorkerRole.HARVESTER,
        ]

        hands_actions: List[List[Any]] = []
        for idx, hand_pos in enumerate(state.hands_pos):
            hand_inv = (
                state.inventories[idx + 1]
                if idx + 1 < len(state.inventories)
                else []
            )
            # Assign role sesuai index hand, fallback ke PLANTER jika > index 2
            assigned_role = hand_roles[idx] if idx < len(hand_roles) else WorkerRole.PLANTER

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