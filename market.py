"""Market action planning and financial resource allocation."""

from __future__ import annotations

from typing import Any, List, Union
from kaggriculture.state import FarmState

MarketOrder = List[Union[str, int]]


class MarketPlanner:
    """Manages market queues, seed buying, land acquisition, and hiring."""

    MAX_ORDERS_PER_TURN: int = 10

    def plan_orders(self, state: FarmState) -> List[MarketOrder]:
        """Formulates ordered list of market actions for the turn."""
        orders: List[MarketOrder] = []
        money = state.money
        unlocked = state.unlocked_quadrants
        seeds = state.seeds
        shed = state.shed

        # 1. Automatic Land Expansion
        if "NE" not in unlocked and money >= 1200:
            orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and money >= 2500 and "NE" in unlocked:
            orders.append(["BUY_LAND"])
            money -= 2000

        # 2. Sell Inventory in Shed
        for item, count in shed.items():
            if count > 0 and item != "WHEAT_SEED":
                orders.append(["SELL", item, count])

        # 3. Dynamic Seed Procurement Strategy
        total_wheat_seeds = seeds.get("WHEAT", 0)
        total_carrot_seeds = seeds.get("CARROT", 0)

        if state.day < 10:
            if total_wheat_seeds < 10 and money >= 100:
                buy_amount = min(15, int(money // 10))
                if buy_amount > 0:
                    orders.append(["BUY_SEED", "WHEAT", buy_amount])
                    money -= buy_amount * 10
        elif state.day < 22:
            if total_carrot_seeds < 10 and money >= 200:
                buy_amount = min(10, int(money // 20))
                if buy_amount > 0:
                    orders.append(["BUY_SEED", "CARROT", buy_amount])
                    money -= buy_amount * 20
            if seeds.get("MELON", 0) < 5 and money >= 400:
                buy_amount = min(5, int(money // 80))
                if buy_amount > 0:
                    orders.append(["BUY_SEED", "MELON", buy_amount])
                    money -= buy_amount * 80

        # 4. Hire Farm Hands during peak production periods
        if state.day >= 3 and state.hour == 0 and state.hires_today == 0 and money >= 500:
            orders.append(["HIRE"])

        return orders[: self.MAX_ORDERS_PER_TURN]