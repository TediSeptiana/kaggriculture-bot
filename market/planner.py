"""Facade planner orchestrating specialized market sub-managers."""

from __future__ import annotations

from typing import Any, List, Union
from market.hiring import HiringManager
from market.land import LandManager
from market.sales import SalesManager
from market.seeds import SeedManager
from state import FarmState

MarketOrder = List[Union[str, int]]


class MarketPlanner:
    """Facade orchestrator coordinating labor, sales, land, and seed managers."""

    MAX_ORDERS_PER_TURN: int = 10
    TOTAL_SEASON_DAYS: int = 30

    def __init__(self, emergency_reserve: float = 100.0) -> None:
        self.emergency_reserve = emergency_reserve
        self.hiring_manager = HiringManager()
        self.sales_manager = SalesManager()
        self.land_manager = LandManager()
        self.seed_manager = SeedManager()

    def get_disposable_cash(self, state: FarmState) -> float:
        """Calculates available liquid capital above safety reserve buffer."""
        reserve = self.emergency_reserve
        if state.hires_today < HiringManager.TARGET_EARLY_HANDS and state.day < self.TOTAL_SEASON_DAYS - 2:
            reserve += 10.0
        return max(0.0, state.money - reserve)

    def plan_orders(self, state: FarmState) -> List[MarketOrder]:
        """Formulates ordered list of utility-optimized market orders."""
        orders: List[MarketOrder] = []
        disposable_cash = self.get_disposable_cash(state)

        # 1. Dispatch Labor Hiring Orders (High Priority)
        orders.extend(self.hiring_manager.plan_hiring_orders(state))

        # 2. Dispatch Sales Orders
        orders.extend(self.sales_manager.plan_sales_orders(state))

        # 3. Dispatch Land Expansion Orders
        land_orders, disposable_cash = self.land_manager.plan_land_orders(
            state, disposable_cash
        )
        orders.extend(land_orders)

        # 4. Dispatch Seed Procurement Orders
        orders.extend(self.seed_manager.plan_seed_orders(state, disposable_cash))

        return orders[: self.MAX_ORDERS_PER_TURN]