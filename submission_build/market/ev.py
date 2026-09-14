"""Expected Value engine berdasarkan spec resmi Kaggriculture."""

from __future__ import annotations

import math
from typing import Dict, Optional

from state import CROP_SPECS, CropSpec, FarmState, MARKET_PARAMS, SHOP_DEMAND, TOWN_CENTER_PRODUCTS


class EVEngine:
    TOTAL_TURNS: int = 720
    SHOP_UNLOCK_INTERVAL: int = 3
    MAX_SHOPS: int = 8

    # ------------------------------------------------------------------
    # Price function — sesuai spec
    # ------------------------------------------------------------------
    @staticmethod
    def _shape(func: str, x: float, T: float) -> float:
        """Shape function untuk price curve."""
        if x <= 0:
            return 0.0
        if func == "linear":
            return x / T
        if func == "sq":
            return (x / T) ** 2
        if func == "sqrt":
            return math.sqrt(x / T)
        if func == "log":
            return math.log(1 + x / T)
        if func == "log10":
            return math.log10(1 + x)
        if func == "hinge":
            u = x / T
            return u + 8 * max(0, u - 1) ** 2
        return x / T

    def price_from_inventory(self, product: str, inventory: float) -> float:
        """Hitung harga dari inventory, sesuai formula spec."""
        params = MARKET_PARAMS.get(product)
        if params is None:
            return 25.0

        base = params["base"]
        I0 = params["I0"]
        T = params["T"]

        if inventory < I0:
            # scarcity → harga naik
            func = params["below_func"]
            target = params["below_target"]
            x = I0 - inventory
            f_T = self._shape(func, T, T)
            amp = (target * base) / f_T if f_T > 0 else 0
            price = base + amp * self._shape(func, x, T)
        else:
            # glut → harga turun
            func = params["above_func"]
            target = params["above_target"]
            x = inventory - I0
            f_T = self._shape(func, T, T)
            amp = (target * base) / f_T if f_T > 0 else 0
            price = base - amp * self._shape(func, x, T)

        return max(1.0, round(price))

    # ------------------------------------------------------------------
    # Expected production
    # ------------------------------------------------------------------
    def expected_production(
        self, spec: CropSpec, remaining_days: float
    ) -> float:
        """Total unit yang diharapkan dari 1 tile, selama sisa musim."""
        if remaining_days < spec.first_yield_day:
            return 0.0

        if spec.crop_type == "one_time":
            return float(spec.max_yield)

        interval = max(1, spec.regrow_interval)
        events = 1 + int((remaining_days - spec.first_yield_day) // interval)
        # Cap di max_yield (production count)
        events = min(events, spec.max_yield)
        # Base yield 1/event, dengan fertilizer bisa 2
        return float(events)

    # ------------------------------------------------------------------
    # Expected demand
    # ------------------------------------------------------------------
    def expected_demand(
        self, product: str, state: FarmState, remaining_days: float
    ) -> float:
        """Total demand selama sisa musim dari shops + town center."""
        daily = 0.0

        # Shops
        for shop_name in (state.unlocked_shops or []):
            shop_map = SHOP_DEMAND.get(shop_name, {})
            daily += shop_map.get(product, 0.0)

        # Town center
        if product in TOWN_CENTER_PRODUCTS:
            daily += 1.0

        # Future shops — estimasi
        # Setiap 3 hari, 1 shop unlock, max 8 total
        days_remaining = remaining_days
        current_shop_count = len(state.unlocked_shops or [])
        future_unlocks = min(
            max(0, self.MAX_SHOPS - current_shop_count),
            int(days_remaining / self.SHOP_UNLOCK_INTERVAL),
        )

        # Rata-rata demand per shop untuk produk ini
        if SHOP_DEMAND:
            avg_daily_per_shop = sum(
                s.get(product, 0.0) for s in SHOP_DEMAND.values()
            ) / len(SHOP_DEMAND)
        else:
            avg_daily_per_shop = 0.0

        # Asumsi unlock merata
        for i in range(future_unlocks):
            days_active = days_remaining - (i * self.SHOP_UNLOCK_INTERVAL)
            if days_active > 0:
                daily += 0  # future shops tidak masuk daily, tapi kita hitung total

        # Total = daily × remaining + future shops × avg × days_active
        total = daily * remaining_days

        for i in range(future_unlocks):
            days_active = max(0.0, remaining_days - (i * self.SHOP_UNLOCK_INTERVAL))
            total += avg_daily_per_shop * days_active

        return total

    # ------------------------------------------------------------------
    # Expected price
    # ------------------------------------------------------------------
    def expected_price(
        self,
        product: str,
        state: FarmState,
        our_production: float,
        remaining_days: float,
    ) -> float:
        current_price = float(state.market_prices.get(product, MARKET_PARAMS.get(product, {}).get("base", 25)))
        current_inv = float((state.market_inventory or {}).get(product, 10000))
        future_demand = self.expected_demand(product, state, remaining_days)

        future_inv = max(0.0, current_inv - future_demand + our_production)
        future_price = self.price_from_inventory(product, future_inv)

        # Blend 60% current + 40% future
        return 0.60 * current_price + 0.40 * future_price

    # ------------------------------------------------------------------
    # Total EV
    # ------------------------------------------------------------------
    def calculate_ev(
        self, crop_name: str, state: FarmState, tile_count: int = 1
    ) -> float:
        spec = CROP_SPECS.get(crop_name)
        if spec is None:
            return -999999.0

        remaining_days = state.remaining_days()
        if remaining_days < spec.first_yield_day:
            return -999999.0

        production_per_tile = self.expected_production(spec, remaining_days)
        total_production = production_per_tile * tile_count

        expected_p = self.expected_price(
            spec.product_name, state, total_production, remaining_days
        )

        revenue = total_production * expected_p
        seed_cost = spec.seed_cost * tile_count

        # Opportunity cost: 3% dari revenue
        opportunity = revenue * 0.03

        return revenue - seed_cost - opportunity

    def rank_crops(self, state: FarmState, available_tiles: int = 1):
        results = []
        for crop_name in CROP_SPECS:
            ev = self.calculate_ev(crop_name, state, available_tiles)
            results.append((crop_name, ev))
        results.append(("WAIT", 0.0))
        results.sort(key=lambda x: x[1], reverse=True)
        return results