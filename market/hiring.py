"""Labor hiring strategy and order generation."""

from __future__ import annotations

from typing import Any, List, Union
from state import FarmState

MarketOrder = List[Union[str, int]]


class HiringManager:
    """Manages labor hiring decisions and orders."""

    TARGET_EARLY_HANDS: int = 3
    TOTAL_SEASON_DAYS: int = 30

    def plan_hiring_orders(self, state: FarmState) -> List[MarketOrder]:
        """Generates HIRE market orders based on daily target hand allocation."""
        orders: List[MarketOrder] = []

        # Pada hour 0 di setiap harinya (sebelum akhir musim), penuhi kuota 3 hand
        if state.hour == 0 and state.day < (self.TOTAL_SEASON_DAYS - 2):
            needed_hires = max(0, self.TARGET_EARLY_HANDS - state.hires_today)
            for _ in range(needed_hires):
                orders.append(["HIRE"])

        return orders