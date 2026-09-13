"""Animal procurement logic. Market BUY_ANIMAL orders."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from state import FarmState

MarketOrder = List[Any]


class AnimalManager:
    """Buy geese and cows based on target count and cash availability."""

    ANIMAL_COST: Dict[str, int] = {"GOOSE": 300, "COW": 400, "SHEEP": 500}

    # ----------------------------------------------------------
    # TARGET — diturunkan drastis karena action economy
    # 1 goose = ~5 turn/hari (feed + collect + fertilizer + move)
    # Hari hanya 24 turn, dan masih ada crops yang butuh perhatian
    # ----------------------------------------------------------
    TARGETS: Dict[str, int] = {
        "GOOSE": 2,    # dari 20 → 2
        "COW": 0,      # dari 4 → 0 (fokus goose dulu)
        "SHEEP": 0,
    }

    # Window pembelian — beri waktu animal untuk produksi
    GOOSE_START_DAY: int = 3
    GOOSE_END_DAY: int = 12    # dari 20 → 12, agar ada minimal 18 hari produksi
    COW_START_DAY: int = 8
    COW_END_DAY: int = 12

    # Reserve — jangan habiskan cash untuk animal
    MIN_CASH_RESERVE: float = 800.0   # dari 500 → 800

    # Minimal hari tersisa agar animal bisa balik modal
    # Goose: $300 / ($50 × 0.8) = ~8 hari untuk balik modal
    MIN_DAYS_TO_RECOVER_GOOSE: int = 10
    MIN_DAYS_TO_RECOVER_COW: int = 12

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def _count_live(self, state: FarmState, animal: str) -> int:
        """Count animals: placed + shed + inventory."""
        count = 0
        try:
            for row in state.tiles:
                for t in row:
                    if isinstance(t, dict) and t.get("animal") == animal:
                        count += 1
        except Exception:
            pass
        shed = getattr(state, "shed", {}) or {}
        count += int(shed.get(animal, 0))
        try:
            for inv in getattr(state, "inventories", []) or []:
                if isinstance(inv, dict):
                    count += int(inv.get(animal, 0))
        except Exception:
            pass
        return count

    def _count_structures(self, state: FarmState) -> Tuple[int, int, int, int]:
        """Return (empty_coops, empty_pastures, total_coops, total_pastures)."""
        empty_coops = 0
        empty_pastures = 0
        total_coops = 0
        total_pastures = 0
        try:
            for row in state.tiles:
                for t in row:
                    if isinstance(t, dict):
                        kind = t.get("kind")
                        if kind == "COOP":
                            total_coops += 1
                            if not t.get("animal"):
                                empty_coops += 1
                        elif kind == "PASTURE":
                            total_pastures += 1
                            if not t.get("animal"):
                                empty_pastures += 1
        except Exception:
            pass
        return empty_coops, empty_pastures, total_coops, total_pastures

    def plan_animal_orders(
        self, state: FarmState, disposable_cash: float
    ) -> Tuple[List[MarketOrder], float]:
        """Return (orders, total_cost)."""
        if not self.enabled:
            return [], 0.0

        orders: List[MarketOrder] = []
        total_cost = 0.0

        day = int(getattr(state, "day", 0))
        money = float(getattr(state, "money", 0.0))
        shed = getattr(state, "shed", {}) or {}

        empty_coops, empty_pastures, total_coops, total_pastures = (
            self._count_structures(state)
        )

        # Hari tersisa sampai akhir musim
        total_days = int(getattr(state, "TOTAL_SEASON_DAYS", 30))
        days_left = total_days - day

        # ==========================================
        # GOOSE — butuh empty coop ATAU belum ada coop sama sekali
        # (kalau belum ada coop, worker BUILD akan bangun dulu)
        # ==========================================
        goose_total = self._count_live(state, "GOOSE")
        goose_in_shed = int(shed.get("GOOSE", 0))
        goose_cost = self.ANIMAL_COST["GOOSE"]

        # Boleh beli goose kalau:
        #   - ada empty coop (langsung place), ATAU
        #   - belum ada coop sama sekali (worker akan BUILD_COOP)
        has_place_for_goose = empty_coops > 0 or total_coops == 0

        goose_ok = (
            self.GOOSE_START_DAY <= day <= self.GOOSE_END_DAY
            and goose_in_shed == 0                              # jangan numpuk di shed
            and goose_total < self.TARGETS["GOOSE"]
            and has_place_for_goose
            and days_left >= self.MIN_DAYS_TO_RECOVER_GOOSE     # cukup waktu balik modal
            and money >= goose_cost + self.MIN_CASH_RESERVE
            and disposable_cash >= goose_cost
        )
        if goose_ok:
            orders.append(["BUY_ANIMAL", "GOOSE", 1])
            total_cost += goose_cost
            disposable_cash -= goose_cost

        # ==========================================
        # COW — butuh empty_pasture
        # ==========================================
        cow_total = self._count_live(state, "COW")
        cow_in_shed = int(shed.get("COW", 0))
        cow_cost = self.ANIMAL_COST["COW"]

        cow_ok = (
            self.COW_START_DAY <= day <= self.COW_END_DAY
            and cow_in_shed == 0
            and cow_total < self.TARGETS["COW"]
            and empty_pastures > 0
            and days_left >= self.MIN_DAYS_TO_RECOVER_COW
            and money >= cow_cost + self.MIN_CASH_RESERVE
            and disposable_cash >= cow_cost
        )
        if cow_ok:
            orders.append(["BUY_ANIMAL", "COW", 1])
            total_cost += cow_cost

        # ==========================================
        # SHEEP — disable untuk sekarang
        # ==========================================
        # (kalau mau aktifkan nanti, uncomment blok di bawah)
        # sheep_total = self._count_live(state, "SHEEP")
        # sheep_in_shed = int(shed.get("SHEEP", 0))
        # sheep_cost = self.ANIMAL_COST["SHEEP"]
        # sheep_ok = (
        #     self.COW_START_DAY <= day <= self.COW_END_DAY
        #     and sheep_in_shed == 0
        #     and sheep_total < self.TARGETS["SHEEP"]
        #     and empty_pastures > 0
        #     and days_left >= 14
        #     and money >= sheep_cost + self.MIN_CASH_RESERVE
        #     and disposable_cash >= sheep_cost
        # )
        # if sheep_ok:
        #     orders.append(["BUY_ANIMAL", "SHEEP", 1])
        #     total_cost += sheep_cost

        return orders, total_cost