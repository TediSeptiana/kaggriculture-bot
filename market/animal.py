"""Animal procurement logic. Market BUY_ANIMAL orders."""

from __future__ import annotations

from typing import Any, Dict, List
from state import FarmState

MarketOrder = List[Any]


class AnimalManager:
    """Buy geese based on target count and cash availability."""

    # Config — bisa di-tuning
    GOOSE_TARGET: int = 6
    MIN_CASH_RESERVE: float = 300.0
    ANIMAL_COST: Dict[str, int] = {"GOOSE": 300, "COW": 400, "SHEEP": 500}

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def _count_live(self, state: FarmState, animal: str) -> int:
        """Count animals: di tile (placed) + di shed (belum placed) + inventory."""
        count = 0
        try:
            for row in state.tiles:
                for t in row:
                    if isinstance(t, dict):
                        if t.get("animal") == animal:
                            count += 1
        except Exception:
            pass
        # di shed
        count += state.shed.get(animal, 0)
        # di inventory worker (kalau state simpan)
        try:
            for inv in getattr(state, "inventories", []):
                if isinstance(inv, dict):
                    count += inv.get(animal, 0)
        except Exception:
            pass
        return count

    def plan_animal_orders(
        self, state: FarmState, available_cash: float
    ) -> List[MarketOrder]:
        """Return BUY_ANIMAL orders. Beli 1 goose per turn supaya tidak overshoot."""
        if not self.enabled:
            return []

        # Jangan beli kalau sudah lewat day 18 (tidak cukup waktu panen)
        if state.day >= 18:
            return []

        # Jangan beli di day 0-2 (fokus bangun farm dulu)
        if state.day < 3:
            return []

        orders: List[MarketOrder] = []
        budget = available_cash - self.MIN_CASH_RESERVE
        if budget < self.ANIMAL_COST["GOOSE"]:
            return orders

        # --- GOOSE ---
        goose_count = self._count_live(state, "GOOSE")
        if goose_count < self.GOOSE_TARGET:
            # Beli 1 per turn untuk trigger task BUILD_COOP
            orders.append(["BUY_ANIMAL", "GOOSE", 1])

        return orders