"""Land expansion utility scoring and acquisition manager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from state import FarmState

MarketOrder = List[Union[str, int]]


class LandManager:
    """Evaluates quadrant expansion utility and generates land orders."""

    LAND_COSTS: Dict[str, float] = {
        "NE": 1000.0,
        "SW": 2000.0,
        "SE": 4000.0,
    }

    def calculate_land_score(
        self, state: FarmState, disposable_cash: float
    ) -> Tuple[Optional[str], float]:
        """Calculates utility score and candidate quadrant for land expansion."""
        unlocked = set(state.unlocked_quadrants)
        target_quad: Optional[str] = None

        if "NE" not in unlocked:
            target_quad = "NE"
        elif "SW" not in unlocked:
            target_quad = "SW"
        elif "SE" not in unlocked:
            target_quad = "SE"

        if target_quad is None:
            return None, 0.0

        cost = self.LAND_COSTS.get(target_quad, 99999.0)
        if disposable_cash < cost:
            return target_quad, 0.0

        tiles = state.tiles
        unlocked_pos = state.get_unlocked_tiles()
        occupied_tiles = sum(1 for x, y in unlocked_pos if tiles[y][x] is not None)

        utilization = occupied_tiles / max(1, len(unlocked_pos))
        land_score = utilization * 0.85
        return target_quad, land_score

    def plan_land_orders(
        self, state: FarmState, disposable_cash: float
    ) -> Tuple[List[MarketOrder], float]:
        """Generates BUY_LAND order if utility threshold is met."""
        orders: List[MarketOrder] = []
        target_quad, land_score = self.calculate_land_score(state, disposable_cash)

        if target_quad and land_score > 0.6:
            cost = self.LAND_COSTS[target_quad]
            orders.append(["BUY_LAND"])
            disposable_cash -= cost

        return orders, disposable_cash