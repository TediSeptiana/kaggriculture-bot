"""Product inventory sales and shed pressure management."""

from __future__ import annotations

from typing import List, Union
from state import FarmState

MarketOrder = List[Union[str, int]]


class SalesManager:
    """Manages shed inventory sales and market liquidation."""

    # ---- Konfigurasi ----
    TOTAL_SEASON_DAYS: int = 30
    SHED_CAPACITY: int = 100

    # Produk yang boleh di-hold sebentar (harga cenderung naik)
    HOLD_PRODUCTS = {"EGG", "MILK", "WOOL"}
    HOLD_MAX_UNITS: int = 3          # jangan hold lebih dari ini

    # Liquidasi paksa di akhir musim
    LIQUIDATE_DAY: int = 28

    # Simpan wheat untuk pakan animal
    WHEAT_FEED_RESERVE_MIN: int = 15
    WHEAT_FEED_RESERVE_PER_ANIMAL: int = 3

    # Barang yang tidak boleh dijual (animal hidup)
    NEVER_SELL = {"GOOSE", "COW", "SHEEP"}

    # Produk yang tidak pernah dijual (benih)
    SEED_SUFFIX = "_SEED"

    def _count_live_animals(self, state: FarmState) -> int:
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
        animal_count = self._count_live_animals(state)
        return max(
            self.WHEAT_FEED_RESERVE_MIN,
            animal_count * self.WHEAT_FEED_RESERVE_PER_ANIMAL,
        )

    def _shed_pressure(self, state: FarmState) -> float:
        total = sum(v for k, v in state.shed.items() if not k.endswith(self.SEED_SUFFIX))
        return total / self.SHED_CAPACITY

    def plan_sales_orders(self, state: FarmState) -> List[MarketOrder]:
        orders: List[MarketOrder] = []

        force_sell_all = state.day >= self.LIQUIDATE_DAY
        shed_pressure = self._shed_pressure(state)
        wheat_reserve = self._wheat_reserve(state)
        near_end = state.day >= self.LIQUIDATE_DAY - 2

        for item, count in state.shed.items():
            if count <= 0:
                continue
            if item.endswith(self.SEED_SUFFIX):
                continue
            if item in self.NEVER_SELL:
                continue

            item_price = state.market_prices.get(item, 0.0)

            # ----------------------------------------------------------
            # Aturan 1: liquidasi paksa di 2 hari terakhir
            # Jual SEMUA termasuk WHEAT reserve — season berakhir
            # ----------------------------------------------------------
            if force_sell_all:
                orders.append(["SELL", item, count])
                continue

            # ----------------------------------------------------------
            # Aturan 2: WHEAT — sisakan untuk pakan HANYA jika masih ada waktu
            # ----------------------------------------------------------
            if item == "WHEAT":
                if near_end:
                    sell_qty = count
                else:
                    sell_qty = max(0, count - wheat_reserve)
                if sell_qty > 0 and item_price > 0:
                    orders.append(["SELL", item, sell_qty])
                continue

            # ----------------------------------------------------------
            # Aturan 3: HOLD_PRODUCTS (EGG/MILK/WOOL)
            # Jual jika: sudah menumpuk >= HOLD_MAX_UNITS, ATAU
            #            shed hampir penuh, ATAU mendekati akhir musim
            # ----------------------------------------------------------
            if item in self.HOLD_PRODUCTS:
                should_sell = (
                    count >= self.HOLD_MAX_UNITS
                    or shed_pressure > 0.7
                    or near_end
                    or item_price <= 0
                )
                if should_sell:
                    orders.append(["SELL", item, count])
                continue

            # ----------------------------------------------------------
            # Aturan 4: produk biasa (CARROT, MELON, dll)
            # Jual kalau ada harga, atau jika shed hampir penuh
            # ----------------------------------------------------------
            if item_price > 0 or shed_pressure > 0.7:
                orders.append(["SELL", item, count])

        return orders