"""Product inventory sales and shed pressure management."""

from __future__ import annotations

from typing import Any, List, Union
from state import FarmState

MarketOrder = List[Union[str, int]]


class SalesManager:
    """Manages shed inventory sales and market liquidation."""

    SHED_CAPACITY: int = 100

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
            if item_price > 0 or sell_urgency_threshold > 0:
                orders.append(["SELL", item, count])

        return orders