"""Utility-driven market action planning and financial resource allocation."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union
from state import CROP_SPECS, CropSpec, FarmState

MarketOrder = List[Union[str, int]]


class MarketPlanner:
    """Utility-based market queue manager for seeds, land, hiring, and sales."""

    MAX_ORDERS_PER_TURN: int = 10
    SHED_CAPACITY: int = 100
    TOTAL_SEASON_DAYS: int = 30

    # Land Purchase Costs per Quadrant Expansion
    LAND_COSTS: Dict[str, float] = {
        "NE": 1000.0,
        "SW": 2000.0,
        "SE": 4000.0,
    }

    def __init__(
        self,
        emergency_reserve: float = 200.0,
        alpha_ppd: float = 0.40,
        beta_roi: float = 0.20,
        gamma_urgency: float = 0.30,
        delta_mkt: float = 0.10,
    ) -> None:
        self.emergency_reserve = emergency_reserve
        self.alpha_ppd = alpha_ppd
        self.beta_roi = beta_roi
        self.gamma_urgency = gamma_urgency
        self.delta_mkt = delta_mkt

    def get_disposable_cash(self, state: FarmState) -> float:
        """Calculates available liquid capital above safety reserve buffer."""
        reserve = self.emergency_reserve
        # Reserve cost for next day's potential hiring
        if state.hires_today == 0 and state.day < self.TOTAL_SEASON_DAYS - 2:
            reserve += 100.0
        return max(0.0, state.money - reserve)

    def calculate_seed_score(
        self, crop_name: str, spec: CropSpec, state: FarmState
    ) -> float:
        """Computes multi-attribute utility score for purchasing a specific seed type."""
        price = state.market_prices.get(spec.product_name, 0.0)
        cost = float(spec.seed_cost)

        # 1. Expected Profit & Profit Per Day (PPD)
        expected_yield = (spec.base_yield + spec.max_yield) / 2.0
        revenue = price * expected_yield
        profit = revenue - cost
        roi = profit / max(1.0, cost)

        days_to_harvest = max(1, spec.first_yield_day)
        ppd = profit / days_to_harvest

        # Normalize PPD against highest market price benchmark
        max_market_price = max(state.market_prices.values(), default=100.0)
        norm_ppd = min(1.0, max(0.0, ppd / (max_market_price * 0.5)))
        norm_roi = min(1.0, max(0.0, roi / 5.0))

        # 2. Seed Deficit Urgency
        current_seeds = state.seeds.get(crop_name, 0)
        unlocked_tiles = len(state.get_unlocked_tiles())
        target_seed_inventory = max(5, unlocked_tiles // 2)

        deficit = max(0, target_seed_inventory - current_seeds)
        urgency = min(1.0, deficit / max(1, target_seed_inventory))

        # 3. Market Relative Value
        market_val = min(1.0, price / max(1.0, max_market_price))

        # Lifecycle constraint: Do not buy slow seeds near end of season
        remaining_days = self.TOTAL_SEASON_DAYS - state.day
        if spec.first_yield_day >= remaining_days:
            return -1.0

        score = (
            self.alpha_ppd * norm_ppd
            + self.beta_roi * norm_roi
            + self.gamma_urgency * urgency
            + self.delta_mkt * market_val
        )
        return score

    def calculate_land_score(self, state: FarmState) -> Tuple[Optional[str], float]:
        """Calculates utility score and candidate quadrant for land expansion."""
        unlocked = set(state.unlocked_quadrants)
        target_quad: Optional[str] = None

        if "NE" not in unlocked:
            target_quad = "NE"
        elif "SW" not in unlocked:
            target_quad = "SW"
        elif "SE" not in unlocked:
            target_quad = "SE"

        if target_quad is None:
            return None, 0.0

        cost = self.LAND_COSTS.get(target_quad, 99999.0)
        disposable_cash = self.get_disposable_cash(state)

        if disposable_cash < cost:
            return target_quad, 0.0

        # Lahan dinilai tinggi jika utilisasi tile produktif saat ini sudah padat
        tiles = state.tiles
        unlocked_pos = state.get_unlocked_tiles()
        occupied_tiles = 0

        for x, y in unlocked_pos:
            t = tiles[y][x]
            if t is not None:
                occupied_tiles += 1

        utilization = occupied_tiles / max(1, len(unlocked_pos))
        land_score = utilization * 0.85
        return target_quad, land_score

    def plan_orders(self, state: FarmState) -> List[MarketOrder]:
        """Formulates ordered list of utility-optimized market orders."""
        orders: List[MarketOrder] = []
        disposable_cash = self.get_disposable_cash(state)

        # 1. Product Sales Logic with Shed Pressure & Reserve Protection
        total_shed_items = sum(state.shed.values())
        shed_pressure = total_shed_items / float(self.SHED_CAPACITY)
        sell_urgency_threshold = 0.2 if shed_pressure > 0.7 else 0.0

        for item, count in state.shed.items():
            if count <= 0 or item.endswith("_SEED"):
                continue

            # Menjual item dari shed
            item_price = state.market_prices.get(item, 0.0)
            if item_price > 0 or sell_urgency_threshold > 0:
                orders.append(["SELL", item, count])

        # 2. Land Expansion Evaluation
        target_quad, land_score = self.calculate_land_score(state)
        if target_quad and land_score > 0.6:
            cost = self.LAND_COSTS[target_quad]
            orders.append(["BUY_LAND"])
            disposable_cash -= cost

        # 3. Dynamic Seed Procurement Evaluation
        seed_scores: List[Tuple[float, str, CropSpec]] = []
        for crop_name, spec in CROP_SPECS.items():
            score = self.calculate_seed_score(crop_name, spec, state)
            if score > 0.0:
                seed_scores.append((score, crop_name, spec))

        # Sort seed berdasarkan utility score tertinggi
        seed_scores.sort(key=lambda x: x[0], reverse=True)

        for _, crop_name, spec in seed_scores:
            if disposable_cash < spec.seed_cost:
                continue

            current_count = state.seeds.get(crop_name, 0)
            target_count = 10 if spec.crop_type == "one_time" else 4
            deficit = max(0, target_count - current_count)

            if deficit > 0:
                affordable = int(disposable_cash // spec.seed_cost)
                buy_amount = min(deficit, affordable, 15)

                if buy_amount > 0:
                    orders.append(["BUY_SEED", crop_name, buy_amount])
                    disposable_cash -= buy_amount * spec.seed_cost

        # 4. Labor Hiring Decision
        if (
            state.day >= 3
            and state.hour == 0
            and state.hires_today == 0
            and disposable_cash >= 200.0
        ):
            # Evaluasi pekerjaan aktif (tanaman unwatered atau weed)
            unlocked_pos = state.get_unlocked_tiles()
            active_work = 0
            for x, y in unlocked_pos:
                t = state.tiles[y][x]
                if isinstance(t, dict):
                    if t.get("kind") == "WEED" or (
                        t.get("kind") == "PLANT" and not t.get("watered_today")
                    ):
                        active_work += 1

            if active_work >= 5:
                orders.append(["HIRE"])

        return orders[: self.MAX_ORDERS_PER_TURN]