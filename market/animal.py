"""Animal procurement logic. Market BUY_ANIMAL orders."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from state import FarmState

MarketOrder = List[Any]


class AnimalManager:
    """Buy geese and cows based on target count and cash availability."""

    ANIMAL_COST: Dict[str, int] = {"GOOSE": 300, "COW": 400, "SHEEP": 500}

    TARGETS: Dict[str, int] = {"GOOSE": 20, "COW": 4, "SHEEP": 0}

    GOOSE_START_DAY: int = 3
    GOOSE_END_DAY: int = 20
    COW_START_DAY: int = 8
    COW_END_DAY: int = 20

    MIN_CASH_RESERVE: float = 500.0

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

    def _count_empty_structures(self, state: FarmState) -> Tuple[int, int]:
        empty_coops = 0
        empty_pastures = 0
        try:
            for row in state.tiles:
                for t in row:
                    if isinstance(t, dict) and not t.get("animal"):
                        kind = t.get("kind")
                        if kind == "COOP":
                            empty_coops += 1
                        elif kind == "PASTURE":
                            empty_pastures += 1
        except Exception:
            pass
        return empty_coops, empty_pastures

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

        empty_coops, empty_pastures = self._count_empty_structures(state)

        # ==========================================
        # GOOSE
        # ==========================================
        goose_total = self._count_live(state, "GOOSE")
        goose_in_shed = int(shed.get("GOOSE", 0))
        goose_cost = self.ANIMAL_COST["GOOSE"]

        goose_ok = (
            self.GOOSE_START_DAY <= day <= self.GOOSE_END_DAY
            and goose_in_shed == 0
            and goose_total < self.TARGETS["GOOSE"]
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
            and money >= cow_cost + self.MIN_CASH_RESERVE
            and disposable_cash >= cow_cost
        )
        if cow_ok:
            orders.append(["BUY_ANIMAL", "COW", 1])
            total_cost += cow_cost

        return orders, total_cost