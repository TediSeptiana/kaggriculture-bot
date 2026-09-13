"""Seed procurement utility scoring and market purchasing."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
from state import CROP_SPECS, CropSpec, FarmState

MarketOrder = List[Union[str, int]]


class SeedManager:
    """Evaluates crop economics and handles seed purchasing orders."""

    TOTAL_SEASON_DAYS: int = 30
    
    # FIX: safety reserve - jangan pernah habiskan cash
    MIN_CASH_RESERVE: float = 200.0
    # FIX: max seed di inventory (buffer 5 di atas kapasitas tile)
    SEED_BUFFER: int = 5
    # FIX: max beli per turn per crop (jangan 15)
    MAX_BUY_PER_TURN: int = 3

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

    # ------------------------------------------------------------------
    # FIX: helper untuk hitung kapasitas
    # ------------------------------------------------------------------
    def _count_empty_tiles(self, state: FarmState) -> int:
        """Hitung tile kosong yang bisa ditanam (None, unlocked, bukan weed)."""
        try:
            tiles = state.tiles
            count = 0
            for row in tiles:
                for t in row:
                    if t is None:
                        count += 1
            return count
        except Exception:
            # fallback
            return len(state.get_unlocked_tiles())

    def _count_animals(self, state: FarmState) -> int:
        """Hitung jumlah hewan hidup (untuk hitung kebutuhan wheat pakan)."""
        try:
            tiles = state.tiles
            count = 0
            for row in tiles:
                for t in row:
                    if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
                        if t.get("animal"):
                            count += 1
            return count
        except Exception:
            return 0

    def plan_seed_orders(
        self, state: FarmState, disposable_cash: float
    ) -> List[MarketOrder]:
        """Generates sorted list of BUY_SEED orders based on utility scores."""
        orders: List[MarketOrder] = []

        # FIX: hanya beli seed di turn pertama setiap hari (hour == 0)
        # Ini mencegah beli 3 seed × 24 turn = 72 seed/hari
        if state.hour != 0:
            return orders

        # FIX 1: reserve cash dulu
        available = max(0.0, disposable_cash - self.MIN_CASH_RESERVE)
        if available < 10.0:
            return orders

        # FIX 2: hitung kapasitas
        empty_tiles = self._count_empty_tiles(state)
        seeds_in_inv = sum(state.seeds.values())

        # FIX 3: kalau seed sudah melebihi kapasitas + buffer, JANGAN BELI
        if seeds_in_inv >= empty_tiles + self.SEED_BUFFER:
            return orders

        # FIX 4: room untuk seed baru
        room_for_seeds = max(0, empty_tiles - seeds_in_inv)
        if room_for_seeds <= 0:
            return orders

        # FIX 5: stop beli kalau late game (hewan lebih penting)
        if state.day >= 18:
            return orders

        seed_scores: List[Tuple[float, str, CropSpec]] = []

        for crop_name, spec in CROP_SPECS.items():
            score = self.calculate_seed_score(crop_name, spec, state)
            if score > 0.0:
                seed_scores.append((score, crop_name, spec))

        seed_scores.sort(key=lambda x: x[0], reverse=True)

        for _, crop_name, spec in seed_scores:
            if available < spec.seed_cost:
                break  # FIX: break bukan continue, karena sorted by score

            current_count = state.seeds.get(crop_name, 0)

            # FIX 6: target konservatif
            if crop_name == "WHEAT":
                # wheat HANYA untuk pakan animal
                animal_count = self._count_animals(state)
                # Simpan 3 hari pakan di depan
                target_count = max(10, animal_count * 3)
            elif crop_name in {"STRAWBERRY", "TOMATO"}:
                # skip dulu, ROI terlalu lambat untuk 30 hari
                target_count = 0
            elif crop_name == "MELON":
                # melon = profit utama, tapi jangan banyak
                target_count = max(5, room_for_seeds // 3)
            elif crop_name == "CARROT":
                target_count = max(5, room_for_seeds // 4)
            else:
                target_count = max(5, room_for_seeds // 5)

            deficit = max(0, target_count - current_count)
            if deficit <= 0:
                continue

            # FIX 7: cap buy per turn KECIL
            affordable = int(available // spec.seed_cost)
            buy_amount = min(deficit, affordable, self.MAX_BUY_PER_TURN, room_for_seeds)

            if buy_amount > 0:
                orders.append(["BUY_SEED", crop_name, buy_amount])
                available -= buy_amount * spec.seed_cost
                room_for_seeds -= buy_amount

                if room_for_seeds <= 0:
                    break

        return orders