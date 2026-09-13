"""Facade planner orchestrating specialized market sub-managers with dynamic cash tracking."""

from __future__ import annotations

from typing import Any, List, Union
from market.animal import AnimalManager
from market.hiring import HiringManager, get_fibonacci_cost
from market.land import LandManager
from market.sales import SalesManager
from market.seeds import SeedManager
from state import FarmState

MarketOrder = List[Union[str, int]]


class MarketPlanner:
    """Facade orchestrator coordinating labor, sales, land, seed, and animal managers."""

    MAX_ORDERS_PER_TURN: int = 10
    TOTAL_SEASON_DAYS: int = 30

    def __init__(self, emergency_reserve: float = 300.0) -> None:
        self.emergency_reserve = emergency_reserve
        self.hiring_manager = HiringManager()
        self.sales_manager = SalesManager()
        self.land_manager = LandManager()
        self.seed_manager = SeedManager()
        self.animal_manager = AnimalManager()

    def get_disposable_cash(self, state: FarmState) -> float:
        """Calculates available liquid capital above the safety reserve buffer."""
        spending_cap = state.money * (0.40 if state.day == 0 else 0.70)
        return max(0.0, min(state.money - self.emergency_reserve, spending_cap))

    def plan_orders(self, state: FarmState) -> List[MarketOrder]:
        """Formulates an ordered list of utility-optimized market orders."""
        orders: List[MarketOrder] = []
        disposable_cash = self.get_disposable_cash(state)

        # ------------------------------------------------------------------
        # 1. LAND EXPANSION (phase-gated)
        # ------------------------------------------------------------------
        if state.day <= 15:
            target_quad = ("NE", 1000.0, 3), ("SW", 2000.0, 7), ("SE", 4000.0, 11)
            for quadrant, cost, deadline in target_quad:
                if quadrant not in state.unlocked_quadrants and state.day >= deadline:
                    # FIX: butuh buffer 2x cost supaya tidak stranded
                    if state.money >= cost * 2 + self.emergency_reserve:
                        orders.append(["BUY_LAND"])
                        disposable_cash -= cost
                    break

        # ------------------------------------------------------------------
        # 2. ANIMAL PURCHASES (delegated to AnimalManager)
        # ------------------------------------------------------------------
        animal_orders, animal_cost = self.animal_manager.plan_animal_orders(
            state, disposable_cash
        )
        orders.extend(animal_orders)
        disposable_cash = max(0.0, disposable_cash - animal_cost)

        # ------------------------------------------------------------------
        # 3. HIRING
        # ------------------------------------------------------------------
        hiring_orders = self.hiring_manager.plan_hiring_orders(state)
        orders.extend(hiring_orders)

        current_hires = state.hires_today
        for _ in hiring_orders:
            current_hires += 1
            cost = get_fibonacci_cost(
                current_hires, self.hiring_manager.farm_hand_cost_mult
            )
            disposable_cash = max(0.0, disposable_cash - cost)

        # ------------------------------------------------------------------
        # 4. SALES (liquidate produce)
        # ------------------------------------------------------------------
        sales_orders = self.sales_manager.plan_sales_orders(state)
        orders.extend(sales_orders)

        sales_proceeds = sum(
            state.market_prices.get(item, 0.0) * count
            for item, count in state.shed.items()
            if count > 0 and not item.endswith("_SEED")
        )
        disposable_cash += sales_proceeds

        # ------------------------------------------------------------------
        # 5. LAND (fallback)
        # ------------------------------------------------------------------
        if not any(order == ["BUY_LAND"] for order in orders):
            land_orders, disposable_cash = self.land_manager.plan_land_orders(
                state, disposable_cash
            )
            orders.extend(land_orders)

        # ------------------------------------------------------------------
        # 6. SEEDS
        # ------------------------------------------------------------------
        seed_orders = self.seed_manager.plan_seed_orders(state, disposable_cash)
        orders.extend(seed_orders)

        return orders[: self.MAX_ORDERS_PER_TURN]