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

    MAX_ORDERS_PER_TURN: int = 25
    TOTAL_SEASON_DAYS: int = 30

    # FIX: Target sisa uang 100-200 coin. Ambil titik tengah 150.0 untuk agresivitas maksimal.
    DEFAULT_EMERGENCY_RESERVE: float = 150.0

    MIN_LAND_UTILIZATION: float = 0.65
    MIN_LAND_CASH_MULTIPLIER: float = 2.0
    MIN_CASH_FOR_LAND: float = 1500.0

    def __init__(self, emergency_reserve: float = DEFAULT_EMERGENCY_RESERVE) -> None:
        self.emergency_reserve = emergency_reserve
        self.hiring_manager = HiringManager()
        self.sales_manager = SalesManager()
        self.land_manager = LandManager()
        self.seed_manager = SeedManager()
        self.animal_manager = AnimalManager()

    def get_disposable_cash(self, state: FarmState) -> float:
        """
        FIX: Agresif maksimal. Sisakan hanya ~150 coin.
        Hari ke-0: gunakan 100% dari kelebihan reserve.
        Hari selanjutnya: gunakan 95% untuk sedikit ruang napas.
        """
        reserve = 150.0
        if state.money <= reserve:
            return 0.0
        
        multiplier = 1.0 if state.day == 0 else 0.95
        return max(0.0, (state.money - reserve) * multiplier)

    def _land_utilization(self, state: FarmState) -> float:
        """Hitung utilisasi tile (used / total unlocked)."""
        try:
            used = 0
            total = 0
            for row in state.tiles:
                for t in row:
                    if isinstance(t, str) and t == "LOCKED":
                        continue
                    total += 1
                    if t is not None:
                        used += 1
            if total == 0:
                return 1.0
            return used / total
        except Exception:
            return 1.0

    def plan_orders(self, state: FarmState) -> List[MarketOrder]:
        """Formulates an ordered list of utility-optimized market orders."""
        orders: List[MarketOrder] = []
        disposable_cash = self.get_disposable_cash(state)

        # ------------------------------------------------------------------
        # 1. LAND EXPANSION (Prioritas tinggi di awal jika kondisi terpenuhi)
        # ------------------------------------------------------------------
        if state.day <= 18:
            utilization = self._land_utilization(state)

            if (
                state.money >= self.MIN_CASH_FOR_LAND
                and utilization >= self.MIN_LAND_UTILIZATION
            ):
                target_quad = (
                    ("NE", 1000.0, 3),
                    ("SW", 2000.0, 6),
                    ("SE", 4000.0, 9),
                )
                for quadrant, cost, deadline in target_quad:
                    if quadrant not in state.unlocked_quadrants and state.day >= deadline:
                        required = cost * self.MIN_LAND_CASH_MULTIPLIER + self.emergency_reserve
                        if state.money >= required:
                            orders.append(["BUY_LAND"])
                            disposable_cash = max(0.0, disposable_cash - cost)
                        break

        # ------------------------------------------------------------------
        # 2. ANIMAL PURCHASES
        # ------------------------------------------------------------------
        animal_orders, animal_cost = self.animal_manager.plan_animal_orders(
            state, disposable_cash
        )
        orders.extend(animal_orders)
        disposable_cash = max(0.0, disposable_cash - animal_cost)

        # ------------------------------------------------------------------
        # 3. WHEAT FEED (Prioritas #1: animal harus di-feed setiap hari)
        # ------------------------------------------------------------------
        shed = getattr(state, "shed", {}) or {}

        # Hitung total animal: tile + shed + inventory
        animal_count = 0
        for row in state.tiles:
            for t in row:
                if isinstance(t, dict) and t.get("animal"):
                    animal_count += 1
        
        animal_count += int(shed.get("COW", 0))
        animal_count += int(shed.get("SHEEP", 0))
        animal_count += int(shed.get("GOOSE", 0))
        
        for inv in getattr(state, "inventories", []) or []:
            if isinstance(inv, list):
                for entry in inv:
                    if isinstance(entry, dict):
                        animal_count += int(entry.get("COW", 0))
                        animal_count += int(entry.get("SHEEP", 0))
                        animal_count += int(entry.get("GOOSE", 0))

        if animal_count > 0:
            wheat_in_shed = int(shed.get("WHEAT", 0))
            # Buffer 3 hari feed per animal
            need_wheat = max(0, animal_count * 3 - wheat_in_shed)

            if need_wheat > 0:
                wheat_price = state.market_prices.get("WHEAT", 30.0)
                # Beli kalau cash cukup + buffer kecil $150
                if state.money > wheat_price * need_wheat + 150:
                    buy_qty = min(need_wheat, 10)  # Max 10 per turn
                    orders.append(["BUY_PRODUCT", "WHEAT", buy_qty])
                    disposable_cash -= wheat_price * buy_qty

        # ------------------------------------------------------------------
        # 4. HIRING
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
        # 5. SALES
        # ------------------------------------------------------------------
        sales_orders = self.sales_manager.plan_sales_orders(state)
        orders.extend(sales_orders)

        # Simulasi penambahan cash dari penjualan untuk perencanaan turn ini
        sales_proceeds = sum(
            state.market_prices.get(item, 0.0) * count
            for item, count in state.shed.items()
            if count > 0 and not item.endswith("_SEED")
        )
        disposable_cash += sales_proceeds

        # ------------------------------------------------------------------
        # 6. LAND fallback (jika belum dibeli di step 1)
        # ------------------------------------------------------------------
        if not any(order == ["BUY_LAND"] for order in orders):
            utilization = self._land_utilization(state)
            if (
                utilization >= self.MIN_LAND_UTILIZATION
                and state.money >= self.MIN_CASH_FOR_LAND
            ):
                land_orders, disposable_cash = self.land_manager.plan_land_orders(
                    state, disposable_cash
                )
                orders.extend(land_orders)

        # ------------------------------------------------------------------
        # 7. SEEDS
        # ------------------------------------------------------------------
        seed_orders = self.seed_manager.plan_seed_orders(state, disposable_cash)
        orders.extend(seed_orders)

        return orders[: self.MAX_ORDERS_PER_TURN]