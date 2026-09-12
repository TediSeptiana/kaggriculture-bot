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
    ANIMAL_TARGETS = {"GOOSE": 20, "COW": 4, "SHEEP": 0}

    def __init__(self, emergency_reserve: float = 300.0) -> None:
        self.emergency_reserve = emergency_reserve
        self.hiring_manager = HiringManager()
        self.sales_manager = SalesManager()
        self.land_manager = LandManager()
        self.seed_manager = SeedManager()
        self.pending_animal: str | None = None

    def get_disposable_cash(self, state: FarmState) -> float:
        """Calculates available liquid capital above the safety reserve buffer."""
        spending_cap = state.money * (0.40 if state.day == 0 else 0.70)
        return max(0.0, min(state.money - self.emergency_reserve, spending_cap))

    def plan_orders(self, state: FarmState) -> List[MarketOrder]:
        """Formulates an ordered list of utility-optimized market orders."""
        orders: List[MarketOrder] = []
        disposable_cash = self.get_disposable_cash(state)

        # Early expansion compounds production. These orders are intentionally
        # phase-gated so land and animals do not consume the emergency reserve.
        if state.day <= 15:
            target_quad = ("NE", 1000.0, 3), ("SW", 2000.0, 7), ("SE", 4000.0, 11)
            for quadrant, cost, deadline in target_quad:
                if quadrant not in state.unlocked_quadrants and state.day >= deadline:
                    if state.money >= cost + self.emergency_reserve:
                        orders.append(["BUY_LAND"])
                        disposable_cash -= cost
                    break

        # Goose first for daily egg income, then a small milk buffer. Animals
        # are bought only after a matching empty structure exists; BUILD/PLACE
        # are worker actions and must never be put in the market queue.
        animal_counts = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
        empty_coops = 0
        empty_pastures = 0
        for row in state.tiles:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") in {"COOP", "PASTURE"}:
                    kind = tile.get("kind")
                    if not tile.get("animal"):
                        if kind == "COOP":
                            empty_coops += 1
                        else:
                            empty_pastures += 1
                    animal = str(tile.get("animal", "")).upper()
                    if animal in animal_counts:
                        animal_counts[animal] += 1
        animal_counts["GOOSE"] += state.shed.get("GOOSE", 0)
        animal_counts["COW"] += state.shed.get("COW", 0)
        if state.day <= 22:
            if self.pending_animal is None and empty_coops > 0 and state.shed.get("GOOSE", 0) == 0 and animal_counts["GOOSE"] < 20 and disposable_cash >= 400:
                orders.append(["BUY_ANIMAL", "GOOSE", 1])
                disposable_cash -= 400
            elif self.pending_animal is None and 9 <= state.day <= 22 and empty_pastures > 0 and state.shed.get("COW", 0) == 0 and animal_counts["COW"] < 4 and disposable_cash >= 400:
                orders.append(["BUY_ANIMAL", "COW", 1])
                disposable_cash -= 400

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
        if not any(order == ["BUY_LAND"] for order in orders):
            land_orders, disposable_cash = self.land_manager.plan_land_orders(
                state, disposable_cash
            )
            orders.extend(land_orders)

        # 4. Dispatch Seed Procurement Orders
        seed_orders = self.seed_manager.plan_seed_orders(state, disposable_cash)
        orders.extend(seed_orders)

        return orders[: self.MAX_ORDERS_PER_TURN]