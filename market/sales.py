from __future__ import annotations

from typing import List, Union
from state import FarmState

MarketOrder = List[Union[str, int]]

class SalesManager:
    """Manages shed inventory sales and market liquidation."""

    SHED_CAPACITY: int = 100
    HOLD_PRODUCTS = {"EGG", "MILK", "WOOL"}
    LIQUIDATE_DAY = 28

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
        """Wheat yang harus disimpan untuk pakan (buffer 5 hari per animal)."""
        animal_count = self._count_live_animals(state)
        if animal_count == 0:
            return 0  # Anti-Bonkos: jual semua Wheat jika belum ada animal (e.g. D4)
            
        return max(
            self.WHEAT_FEED_RESERVE_MIN,
            animal_count * self.WHEAT_FEED_RESERVE_PER_ANIMAL,
        )

    def plan_sales_orders(self, state: FarmState) -> List[MarketOrder]:
        orders = []
        # Paksa jual semua di 2 hari terakhir
        force_sell_all = state.day >= (30 - 2)  # TOTAL_SEASON_DAYS = 30
        
        # FIX: Hitung reserve di awal untuk mencegah bug arbitrase (jual-beli berulang)
        wheat_reserve = self._wheat_reserve(state)
        
        for item, count in state.shed.items():
            if count <= 0 or item.endswith("_SEED"):
                continue
            if item in ("GOOSE", "COW", "SHEEP"):
                continue  # animal hidup jangan dijual
            
            # ============================================================
            # FIX: Cegah bug arbitrase pada WHEAT
            # ============================================================
            if item == "WHEAT":
                if count <= wheat_reserve:
                    continue  # Simpan untuk pakan, jangan dijual
                
                # Jual HANYA kelebihannya
                sell_qty = count - wheat_reserve
                orders.append(["SELL", "WHEAT", sell_qty])
                continue
            
            # Untuk produk non-WHEAT, jual semua untuk memaksimalkan cash flow agresif
            item_price = state.market_prices.get(item, 0.0)
            
            if force_sell_all:
                orders.append(["SELL", item, count])
            elif item_price > 0:
                orders.append(["SELL", item, count])
                
        return orders