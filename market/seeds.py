"""Seed procurement utility scoring and market purchasing."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
from state import CROP_SPECS, CropSpec, FarmState

MarketOrder = List[Union[str, int]]


class SeedManager:
    """Evaluates crop economics and handles seed purchasing orders."""

    TOTAL_SEASON_DAYS: int = 30

    # Reserve minimum cash
    MIN_CASH_RESERVE: float = 300.0
    # Buffer seed di atas kapasitas tile
    SEED_BUFFER: int = 5
    # Max beli per turn per crop
    MAX_BUY_PER_TURN: int = 20

    # Late-game cutoff: berhenti beli hanya di 2 hari terakhir
    # (CARROT/WHEAT first_yield = 2 hari, masih bisa panen D28-D29)
    LATE_GAME_CUTOFF: int = 25

    # Crop cepat untuk fallback
    FAST_CROPS = {"WHEAT", "CARROT"}

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

        # FIX: kalau harga = 0 (key mismatch), jangan beli
        if price <= 0.0:
            return -1.0

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
        # FIX: gunakan > bukan >= agar crop yang pas muat tetap diizinkan
        # Contoh: CARROT first_yield = 2, remaining = 2 → masih bisa panen D+2
        if spec.first_yield_day > remaining_days:
            return -1.0

        return (
            self.alpha_ppd * norm_ppd
            + self.beta_roi * norm_roi
            + self.gamma_urgency * urgency
            + self.delta_mkt * market_val
        )

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

        # FIX: boleh beli seed setiap turn di D00-D03 untuk agresif
        if state.hour != 0 and state.day > 3:
            return orders

        # Reserve cash dulu
        available = max(0.0, disposable_cash - self.MIN_CASH_RESERVE)
        if available < 10.0:
            return orders

        # Hitung kapasitas
        empty_tiles = self._count_empty_tiles(state)
        seeds_in_inv = sum(state.seeds.values())

        # Kalau seed sudah melebihi kapasitas + buffer, JANGAN BELI
        if seeds_in_inv >= empty_tiles + self.SEED_BUFFER:
            return orders

        room_for_seeds = max(0, empty_tiles - seeds_in_inv)
        if room_for_seeds <= 0:
            return orders

        # FIX: late-game cutoff hanya 2 hari terakhir
        # CARROT/WHEAT masih bisa ditanam sampai D28
        if state.day >= self.LATE_GAME_CUTOFF:
            return orders

        # Hitung sisa hari
        remaining_days = self.TOTAL_SEASON_DAYS - state.day

        seed_scores: List[Tuple[float, str, CropSpec]] = []

        for crop_name, spec in CROP_SPECS.items():
            score = self.calculate_seed_score(crop_name, spec, state)
            if score > 0.0:
                seed_scores.append((score, crop_name, spec))

        # FIX: kalau tidak ada crop yang muat (semua score -1), coba crop cepat
        if not seed_scores:
            for crop_name in self.FAST_CROPS:
                spec = CROP_SPECS.get(crop_name)
                if spec is None:
                    continue
                if spec.first_yield_day <= remaining_days:
                    # score rendah tapi lebih baik daripada tidak tanam
                    seed_scores.append((0.5, crop_name, spec))

        seed_scores.sort(key=lambda x: x[0], reverse=True)

        for _, crop_name, spec in seed_scores:
            if available < spec.seed_cost:
                break

            current_count = state.seeds.get(crop_name, 0)

            # Target konservatif berdasarkan crop
            if crop_name == "WHEAT":
                # FIX: WHEAT D00-D01 boost untuk feed buffer
                if state.day <= 1:
                    target_count = 10
                else:
                    animal_count = self._count_animals(state)
                    target_count = max(4, animal_count * 3) if animal_count > 0 else 4

            elif crop_name in {"STRAWBERRY", "TOMATO"}:
                target_count = 0

            elif crop_name == "MELON":
                if remaining_days < 10:
                    target_count = 0
                else:
                    # FIX: lebih agresif MELON
                    target_count = max(8, min(15, room_for_seeds // 2))

            elif crop_name == "CARROT":
                # FIX: kurangi CARROT — MELON lebih profitable
                target_count = max(3, room_for_seeds // 6)

            else:
                target_count = max(5, room_for_seeds // 5)

            deficit = max(0, target_count - current_count)
            if deficit <= 0:
                continue

            affordable = int(available // spec.seed_cost)
            buy_amount = min(deficit, affordable, self.MAX_BUY_PER_TURN, room_for_seeds)

            if buy_amount > 0:
                orders.append(["BUY_SEED", crop_name, buy_amount])
                available -= buy_amount * spec.seed_cost
                room_for_seeds -= buy_amount

                if room_for_seeds <= 0:
                    break

        return orders