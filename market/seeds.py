"""Seed procurement utility scoring and market purchasing."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
from state import CROP_SPECS, CropSpec, FarmState

MarketOrder = List[Union[str, int]]


class SeedManager:
    """Evaluates crop economics and handles seed purchasing orders."""

    TOTAL_SEASON_DAYS: int = 30

    def __init__(
        self,
        alpha_ppd: float = 0.40,
        beta_roi: float = 0.20,
        gamma_urgency: float = 0.30,
        delta_mkt: float = 0.10,
    ) -> None:
        self.alpha_ppd = alpha_ppd
        self.beta_roi = beta_roi
        self.gamma_urgency = gamma_urgency
        self.delta_mkt = delta_mkt

    def calculate_seed_score(
        self, crop_name: str, spec: CropSpec, state: FarmState
    ) -> float:
        """Computes multi-attribute utility score for purchasing a specific seed type."""
        price = state.market_prices.get(spec.product_name, 0.0)
        cost = float(spec.seed_cost)

        expected_yield = (spec.base_yield + spec.max_yield) / 2.0
        revenue = price * expected_yield
        profit = revenue - cost
        roi = profit / max(1.0, cost)

        days_to_harvest = max(1, spec.first_yield_day)
        ppd = profit / days_to_harvest

        max_market_price = max(state.market_prices.values(), default=100.0)
        norm_ppd = min(1.0, max(0.0, ppd / (max_market_price * 0.5)))
        norm_roi = min(1.0, max(0.0, roi / 5.0))

        current_seeds = state.seeds.get(crop_name, 0)
        unlocked_tiles = len(state.get_unlocked_tiles())
        target_seed_inventory = max(5, unlocked_tiles // 2)

        deficit = max(0, target_seed_inventory - current_seeds)
        urgency = min(1.0, deficit / max(1, target_seed_inventory))

        market_val = min(1.0, price / max(1.0, max_market_price))

        remaining_days = self.TOTAL_SEASON_DAYS - state.day
        if spec.first_yield_day >= remaining_days:
            return -1.0

        return (
            self.alpha_ppd * norm_ppd
            + self.beta_roi * norm_roi
            + self.gamma_urgency * urgency
            + self.delta_mkt * market_val
        )

    def plan_seed_orders(
        self, state: FarmState, disposable_cash: float
    ) -> List[MarketOrder]:
        """Generates sorted list of BUY_SEED orders based on utility scores."""
        orders: List[MarketOrder] = []
        seed_scores: List[Tuple[float, str, CropSpec]] = []

        for crop_name, spec in CROP_SPECS.items():
            score = self.calculate_seed_score(crop_name, spec, state)
            if score > 0.0:
                seed_scores.append((score, crop_name, spec))

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

        return orders