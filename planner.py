"""High-level orchestration agent combining worker and market planners."""

from __future__ import annotations

from typing import Any, Dict, List
from market import MarketPlanner
from state import FarmState
from worker import WorkerPlanner


class AgentPlanner:
    """Orchestrates state parsing and coordinates sub-planners for each turn."""

    def __init__(self) -> None:
        self.worker_planner = WorkerPlanner()
        self.market_planner = MarketPlanner()

    def plan_turn(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point processing observation dict and returning action payload."""
        state = FarmState.from_obs(obs)

        # Market Queue Planning
        market_actions = self.market_planner.plan_orders(state)

        # Farmer Unit Planning
        farmer_inv = state.inventories[0] if len(state.inventories) > 0 else []
        farmer_action = self.worker_planner.decide_action(
            state.farmer_pos, state, farmer_inv
        )

        # Hired Hands Planning
        hands_actions: List[List[Any]] = []
        for idx, hand_pos in enumerate(state.hands_pos):
            hand_inv = (
                state.inventories[idx + 1]
                if idx + 1 < len(state.inventories)
                else []
            )
            hand_act = self.worker_planner.decide_action(hand_pos, state, hand_inv)
            hands_actions.append(hand_act)

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": market_actions,
        }