"""Phase-based daily targets for the Kaggriculture market planner -- Anti-Bonkos Strategy."""

from __future__ import annotations

from typing import Any, Dict

from state import FarmState


class DailyConfig:
    """Return the operational policy for the current season day.

    Anti-Bonkos Phases:
      D0        : BOOTSTRAP     -- 15 Melon + 10 Wheat, min 5 hands
      D1-D3     : PRE_WHEAT     -- water and wait, no new seeds
      D4-D7     : WHEAT_CYCLE_1 -- panen Wheat D4, reinvest, beli Cow D5+
      D8-D9     : WHEAT_CYCLE_2 -- panen Wheat D8, akumulasi cash
      D10-D19   : SCALE_UP      -- 30 Melon + 20 Wheat, beli tanah NE D10
      D20-D22   : BIG_HARVEST   -- panen besar 2, beli SW+SE, 8 cow+8 sheep
      D23-D29   : FINAL_PUSH    -- Melon+Cow(Susu)+Sheep(Wol), liquidate
    """

    PHASES = (
        (0, 0, "BOOTSTRAP"),
        (1, 3, "PRE_WHEAT"),
        (4, 7, "WHEAT_CYCLE_1"),
        (8, 9, "WHEAT_CYCLE_2"),
        (10, 19, "SCALE_UP"),
        (20, 22, "BIG_HARVEST"),
        (23, 29, "FINAL_PUSH"),
    )

    @classmethod
    def get_config(cls, state: FarmState) -> Dict[str, Any]:
        day = state.day
        config: Dict[str, Any] = {
            "phase": "FINAL_PUSH" if day >= 23 else "UNKNOWN",
            "max_hands": 8,
            "target_cow": 0,
            "target_sheep": 0,
            "seed_targets": {},
            "seed_daily_caps": {},
            "buy_land": False,
            "sell_batch_size": 25,
            "stop_seed": [],
            "min_cash_buffer": 0,
            "stop_animal": day >= 25,
            "force_sell_all": day >= 28,
        }

        if day == 0:
            # BOOTSTRAP: Modal 3000 → beli 15 Melon (1200) + 10 Wheat (100)
            # Hire 1 hand, sisa ~1550 coin. Fibonacci cost sangat murah di awal.
            config.update(
                phase="BOOTSTRAP",
                max_hands=5,
                target_cow=0,
                target_sheep=0,
                seed_targets={"MELON": 15, "WHEAT": 10},
                min_cash_buffer=150,
                sell_batch_size=0,
                buy_land=False,
            )
        elif day <= 3:
            # PRE_WHEAT: Tunggu Wheat & Melon tumbuh, jaga action capacity
            config.update(
                phase="PRE_WHEAT",
                max_hands=5,
                target_cow=0,
                target_sheep=0,
                seed_daily_caps={"WHEAT": 5},
                min_cash_buffer=200,
                sell_batch_size=0,
                buy_land=False,
            )
        elif day <= 7:
            # WHEAT_CYCLE_1: Panen Wheat D4 (+1000), reinvest, beli Cow mulai D5
            config.update(
                phase="WHEAT_CYCLE_1",
                max_hands=6,
                target_cow=1,
                target_sheep=0,
                seed_daily_caps={"WHEAT": 10, "MELON": 5},
                min_cash_buffer=300,
                sell_batch_size=5,
                buy_land=False,
            )
        elif day <= 9:
            # WHEAT_CYCLE_2: Panen Wheat D8 (+1000), akumulasi cash untuk panen besar D10
            config.update(
                phase="WHEAT_CYCLE_2",
                max_hands=6,
                target_cow=2,
                target_sheep=1,
                seed_daily_caps={"WHEAT": 5},
                min_cash_buffer=500,
                sell_batch_size=10,
                buy_land=False,
            )
        elif day <= 19:
            # SCALE_UP: Beli tanah NE D10 (-1000), 30 Melon + 20 Wheat di 50 tiles
            # Modal post-D10 harvest: ~25K. 3+ workers untuk 50 tiles.
            config.update(
                phase="SCALE_UP",
                max_hands=7,
                target_cow=5,
                target_sheep=3,
                seed_targets={"MELON": 30, "WHEAT": 20} if day == 10 else {},
                seed_daily_caps={"WHEAT": 10, "MELON": 8},
                buy_land=day >= 10,
                min_cash_buffer=1000,
                sell_batch_size=10,
            )
        elif day <= 22:
            # BIG_HARVEST: Panen besar 2 D20 (~45K Melon + 2K Wheat)
            # Beli SW (-2000) + SE (-4000). Full 100 tiles. 8 cow + 8 sheep agresif.
            config.update(
                phase="BIG_HARVEST",
                max_hands=8,
                target_cow=8,
                target_sheep=8,
                seed_targets={"MELON": 20, "WHEAT": 20} if day == 20 else {},
                seed_daily_caps={"WHEAT": 15, "MELON": 15},
                buy_land=True,
                min_cash_buffer=2000,
                sell_batch_size=15,
            )
        else:
            # FINAL_PUSH D23-D29: Melon + Cow (Susu) + Sheep (Wol) mix
            # Target: Susu 8 cow * ~96K + Wol 8 sheep * ~40K = 200K+
            config.update(
                phase="FINAL_PUSH",
                max_hands=max(2, 8 - max(0, day - 26)),
                target_cow=8,
                target_sheep=8,
                seed_daily_caps={"WHEAT": 10} if day <= 26 else {},
                stop_seed=(
                    ["MELON", "STRAWBERRY", "TOMATO", "CARROT"]
                    + (["WHEAT"] if day >= 27 else [])
                ),
                min_cash_buffer=0,
                sell_batch_size=25,
                force_sell_all=day >= 28,
                stop_animal=day >= 25,
            )

        return config


__all__ = ["DailyConfig"]
