"""Animal procurement logic. Market BUY_ANIMAL orders."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from state import FarmState

MarketOrder = List[Any]


class AnimalManager:
    """Buy cows and sheep based on target count and cash availability."""

    ANIMAL_COST: Dict[str, int] = {"GOOSE": 300, "COW": 400, "SHEEP": 500}

    # ============================================================
    # FASE B: Fokus Cow + Sheep (bukan Goose)
    # Target: 3 Cow + 2 Sheep = 5 animal
    # ============================================================
    TARGETS: Dict[str, int] = {
        "GOOSE": 0,
        "COW": 4,
        "SHEEP": 3,
    }

    # Mulai D5 (setelah cash flow stabil), bukan D0
    GOOSE_START_DAY: int = 3
    GOOSE_END_DAY: int = 12
    COW_START_DAY: int = 0
    COW_END_DAY: int = 12
    SHEEP_START_DAY: int = 5
    SHEEP_END_DAY: int = 14

    # Reserve cash minimum agresif di awal
    MIN_CASH_RESERVE: float = 200.0

    # Minimal hari tersisa agar animal balik modal
    MIN_DAYS_TO_RECOVER_GOOSE: int = 10
    MIN_DAYS_TO_RECOVER_COW: int = 12
    MIN_DAYS_TO_RECOVER_SHEEP: int = 10

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

        total_days = int(getattr(state, "TOTAL_SEASON_DAYS", 30))
        days_left = total_days - day

        # ==========================================
        # COW
        # ==========================================
        cow_total = self._count_live(state, "COW")
        cow_in_shed = int(shed.get("COW", 0))
        cow_cost = self.ANIMAL_COST["COW"]

        # Guard rebuy loop — cek apakah ada cow yang masih hidup
        cow_placed = sum(
            1 for row in state.tiles for t in row
            if isinstance(t, dict) and t.get("animal") == "COW"
        )

        # Cek apakah butuh pasture baru
        has_place_for_cow = empty_pastures > 0 or total_pastures == 0

        # Berapa banyak cow yang perlu dibeli berdasarkan total
        cows_needed = self.TARGETS["COW"] - cow_total

        cow_ok = (
            self.COW_START_DAY <= day <= self.COW_END_DAY
            and cow_in_shed == 0  # jangan numpuk di shed
            and cows_needed > 0
            and has_place_for_cow
            and days_left >= self.MIN_DAYS_TO_RECOVER_COW
            and money >= cow_cost + self.MIN_CASH_RESERVE
            and disposable_cash >= cow_cost
        )
        if cow_ok:
            orders.append(["BUY_ANIMAL", "COW", 1])
            total_cost += cow_cost
            disposable_cash -= cow_cost

        # ==========================================
        # SHEEP
        # ==========================================
        sheep_total = self._count_live(state, "SHEEP")
        sheep_in_shed = int(shed.get("SHEEP", 0))
        sheep_cost = self.ANIMAL_COST["SHEEP"]

        sheep_placed = sum(
            1 for row in state.tiles for t in row
            if isinstance(t, dict) and t.get("animal") == "SHEEP"
        )

        has_place_for_sheep = empty_pastures > 0 or total_pastures == 0

        sheeps_needed = self.TARGETS["SHEEP"] - sheep_total

        sheep_ok = (
            self.SHEEP_START_DAY <= day <= self.SHEEP_END_DAY
            and sheep_in_shed == 0
            and sheeps_needed > 0
            and has_place_for_sheep
            and days_left >= self.MIN_DAYS_TO_RECOVER_SHEEP
            and money >= sheep_cost + self.MIN_CASH_RESERVE
            and disposable_cash >= sheep_cost
        )
        if sheep_ok:
            orders.append(["BUY_ANIMAL", "SHEEP", 1])
            total_cost += sheep_cost
            disposable_cash -= sheep_cost

        # ==========================================
        # GOOSE — disabled (target 0)
        # ==========================================

        return orders, total_cost