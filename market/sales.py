"""Product inventory sales and shed pressure management."""

from __future__ import annotations

from typing import Any, List, Union
from state import FarmState

MarketOrder = List[Union[str, int]]


class SalesManager:
    """Manages shed inventory sales and market liquidation."""

    SHED_CAPACITY: int = 100

    HOLD_PRODUCTS = {"EGG", "MILK", "WOOL"}
    LIQUIDATE_DAY = 28
    HOLD_MULTIPLIER = 1.20

    def plan_sales_orders(self, state: FarmState) -> List[MarketOrder]:
        """Generates SELL market orders for items stored in shed."""
        orders: List[MarketOrder] = []
        total_shed_items = sum(state.shed.values())
        shed_pressure = total_shed_items / float(self.SHED_CAPACITY)
        sell_urgency_threshold = 0.2 if shed_pressure > 0.7 else 0.0

        for item, count in state.shed.items():
            if count <= 0 or item.endswith("_SEED"):
                continue
            item_price = state.market_prices.get(item, 0.0)

            # Preserve high-value recurring products while prices are rising.
            # Liquidate before the final day so the shed cannot overflow.
            if item in self.HOLD_PRODUCTS and state.day < self.LIQUIDATE_DAY:
                if shed_pressure <= 0.90:
                    continue
                count = max(1, count - int(self.SHED_CAPACITY * 0.75))

            base_price = {"WHEAT": 25.0, "CARROT": 35.0, "TOMATO": 60.0, "STRAWBERRY": 120.0, "MELON": 250.0}.get(item, 0.0)
            price_is_rising = base_price > 0 and item_price >= base_price * self.HOLD_MULTIPLIER
            if state.day < self.LIQUIDATE_DAY and price_is_rising and shed_pressure <= 0.90:
                continue

            if item_price > 0 or sell_urgency_threshold > 0:
                orders.append(["SELL", item, min(count, 10) if state.day < self.LIQUIDATE_DAY else count])

        return orders