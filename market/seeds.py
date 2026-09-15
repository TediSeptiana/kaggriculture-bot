"""Seed procurement utility scoring and market purchasing."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
from state import CROP_SPECS, CropSpec, FarmState

MarketOrder = List[Union[str, int]]


class SeedManager:
    """Evaluates crop economics and handles seed purchasing orders."""

    TOTAL_SEASON_DAYS: int = 30

    # Reserve minimum cash
    MIN_CASH_RESERVE: float = 50.0
    # Buffer seed di atas kapasitas tile (dinaikkan agar tidak memblokir scale-up agresif)
    SEED_BUFFER: int = 25
    # Max beli per turn per crop
    MAX_BUY_PER_TURN: int = 20

    # Stop stocking seeds once the profitable melon window has closed.
    # Anti-Bonkos: D20-D22 masih tanam Melon + Wheat untuk final push
    LATE_GAME_CUTOFF: int = 23

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
            for y, row in enumerate(tiles):
                for x, t in enumerate(row):
                    if (
                        t is None
                        and state.get_quadrant(x, y) in state.unlocked_quadrants
                    ):
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

        # Anti-Bonkos: beli seed di D00-D03 setiap turn (agresif bootstrap)
        # D10 dan D20 adalah momen scale-up besar, juga beli setiap turn
        if state.hour != 0 and state.day > 3 and state.day not in (10, 20):
            return orders

        # Reserve cash dulu
        available = max(0.0, disposable_cash - self.MIN_CASH_RESERVE)
        if available < 10.0:
            return orders

        # FIX: cegah cash crash D05-D12 yang bikin hands 0
        # Kalau cash di bawah $800, batasi belanja max $200
        if state.money < 800 and state.day >= 5:
            available = min(available, 200.0)

        # FIX: kalau cash < $400, JANGAN beli sama sekali
        if state.money < 400 and state.day >= 5:
            return orders

        # Hitung kapasitas
        empty_tiles = self._count_empty_tiles(state)
        seeds_in_inv = sum(state.seeds.values())

        # Kalau seed sudah melebihi kapasitas + buffer, JANGAN BELI (kecuali D0 untuk initial bootstrap)
        if seeds_in_inv >= empty_tiles + self.SEED_BUFFER and state.day > 0:
            return orders

        room_for_seeds = max(0, empty_tiles + self.SEED_BUFFER - seeds_in_inv)
        if room_for_seeds <= 0 and state.day > 0:
            return orders
        elif state.day == 0:
            room_for_seeds = 999  # Bypass capacity check on D0

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

            # ===== ANTI-BONKOS SEED TARGETS =====
            if crop_name == "WHEAT":
                # Day 0: target 10 Wheat (sesuai rencana Anti-Bonkos)
                # Day 5+: scale berdasarkan jumlah animal untuk pakan
                if state.day == 0:
                    target_count = 10
                elif state.day <= 3:
                    target_count = 10  # pertahankan buffer
                else:
                    animal_count = self._count_animals(state)
                    # Buffer 3 hari per animal + minimal 5
                    target_count = max(5, animal_count * 3)

            elif crop_name in {"STRAWBERRY", "TOMATO"}:
                # Anti-Bonkos: skip Strawberry & Tomato, fokus Melon
                target_count = 0

            elif crop_name == "CARROT":
                # Anti-Bonkos: minimal Carrot, Melon jauh lebih profitable
                # Hanya sebagai fallback jika Melon tidak bisa ditanam
                if remaining_days >= 10:
                    target_count = 0  # skip Carrot jika masih bisa Melon
                else:
                    target_count = max(3, room_for_seeds // 4)

            elif crop_name == "MELON":
                if remaining_days < 10:
                    target_count = 0
                else:
                    # Anti-Bonkos Day 0: target 15 Melon
                    # Day 10: target 30 Melon (scale-up setelah panen besar 1)
                    # Day 20: target 20 Melon (BIG_HARVEST + animal phase)
                    if state.day == 0:
                        target_count = 15
                    elif state.day == 10:
                        target_count = 30
                    elif state.day == 20:
                        target_count = 20
                    else:
                        # Di antara milestone: isi sampai 2/3 kapasitas tile
                        target_count = max(8, min(30, room_for_seeds * 2 // 3))

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