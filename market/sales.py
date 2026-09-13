"""Product inventory sales and shed pressure management."""

from __future__ import annotations

from typing import Any, List, Union
from state import FarmState

MarketOrder = List[Union[str, int]]


class SalesManager:
    """Manages shed inventory sales and market liquidation."""

    SHED_CAPACITY: int = 100

    HOLD_PRODUCTS = {"EGG", "MILK", "WOOL"}
    LIQUIDATE_DAY = 28
    HOLD_MULTIPLIER = 1.20

    # FIX: simpan wheat untuk pakan animal
    WHEAT_FEED_RESERVE_MIN: int = 15
    WHEAT_FEED_RESERVE_PER_ANIMAL: int = 3

    def _count_live_animals(self, state: FarmState) -> int:
        """Hitung animal hidup di tile."""
        count = 0
        try:
            for row in state.tiles:
                for t in row:
                    if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
                        if t.get("animal"):
                            count += 1
        except Exception:
            pass
        return count

    def _wheat_reserve(self, state: FarmState) -> int:
        """Wheat yang harus disimpan untuk pakan."""
        animal_count = self._count_live_animals(state)
        return max(
            self.WHEAT_FEED_RESERVE_MIN,
            animal_count * self.WHEAT_FEED_RESERVE_PER_ANIMAL,
        )

    def plan_sales_orders(self, state: FarmState) -> List[MarketOrder]:
        orders = []
        # Paksa jual semua di 2 hari terakhir
        force_sell_all = state.day >= (self.TOTAL_SEASON_DAYS - 2)
        
        for item, count in state.shed.items():
            if count <= 0 or item.endswith("_SEED"):
                continue
            if item in ("GOOSE", "COW", "SHEEP"):
                continue  # animal hidup jangan dijual
            
            item_price = state.market_prices.get(item, 0.0)
            
            if force_sell_all:
                orders.append(["SELL", item, count])
            elif item_price > 0:
                # Jual semua jika price di atas 80% peak (bisa cek rolling max)
                orders.append(["SELL", item, count])
        return orders