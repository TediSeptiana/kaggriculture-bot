"""Facade planner orchestrating specialized market sub-managers with dynamic cash tracking."""

from __future__ import annotations

from typing import Any, List, Union
from market.hiring import HiringManager, get_fibonacci_cost
from market.land import LandManager
from market.sales import SalesManager
from market.seeds import SeedManager
from state import FarmState

MarketOrder = List[Union[str, int]]


class MarketPlanner:
    """Facade orchestrator coordinating labor, sales, land, and seed managers."""

    MAX_ORDERS_PER_TURN: int = 10
    TOTAL_SEASON_DAYS: int = 30

    def __init__(self, emergency_reserve: float = 300.0) -> None:
        self.emergency_reserve = emergency_reserve
        self.hiring_manager = HiringManager()
        self.sales_manager = SalesManager()
        self.land_manager = LandManager()
        self.seed_manager = SeedManager()

    def get_disposable_cash(self, state: FarmState) -> float:
        """Calculates available liquid capital above the safety reserve buffer."""
        return max(0.0, state.money - self.emergency_reserve)

    def plan_orders(self, state: FarmState) -> List[MarketOrder]:
        """Formulates an ordered list of utility-optimized market orders."""
        orders: List[MarketOrder] = []
        disposable_cash = self.get_disposable_cash(state)

        # 1. Dispatch Labor Hiring Orders (Optimized by MDEV)
        hiring_orders = self.hiring_manager.plan_hiring_orders(state)
        orders.extend(hiring_orders)

        # Deduct actual planned hiring expenditure from disposable cash
        # Evaluasi biaya Fibonacci untuk setiap order HIRE yang dikeluarkan
        current_hires = state.hires_today
        for _ in hiring_orders:
            current_hires += 1
            cost = get_fibonacci_cost(current_hires, self.hiring_manager.farm_hand_cost_mult)
            disposable_cash = max(0.0, disposable_cash - cost)

        # 2. Dispatch Sales Orders (Liquidates produce to increase cash)
        sales_orders = self.sales_manager.plan_sales_orders(state)
        orders.extend(sales_orders)
        # Setelah sales_orders ditambahkan, estimasi proceeds:
        sales_proceeds = sum(
            state.market_prices.get(item, 0.0) * count
            for item, count in state.shed.items()
            if count > 0 and not item.endswith("_SEED")
        )
        disposable_cash += sales_proceeds


        # 3. Dispatch Land Expansion Orders
        land_orders, disposable_cash = self.land_manager.plan_land_orders(
            state, disposable_cash
        )
        orders.extend(land_orders)

        # 4. Dispatch Seed Procurement Orders
        seed_orders = self.seed_manager.plan_seed_orders(state, disposable_cash)
        orders.extend(seed_orders)

        return orders[: self.MAX_ORDERS_PER_TURN]