"""Evaluates unit inventory and immediate tile execution feasibility."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from state import CROP_SPECS, FarmState, Pos, Tile


class TaskEvaluator:
    """Evaluates tile status and returns executable action lists."""

    @staticmethod
    def check_immediate_tile_action(
        unit_pos: Pos,
        state: FarmState,
        inventory: List[Dict[str, Any]],
        role_type: str,
    ) -> Optional[List[Any]]:
        ux, uy = unit_pos
        tiles = state.tiles
        current_tile: Tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None

        def carried(item: str) -> bool:
            return any(
                isinstance(entry, dict)
                and (
                    str(entry.get("item", entry.get("name", ""))).upper() == item
                    or int(entry.get(item, 0)) > 0
                )
                and int(entry.get("count", entry.get("quantity", entry.get(item, 1)))) > 0
                for entry in inventory
            )

        # ============================================================
        # BUILD — HANYA kalau ada animal di shed
        # ============================================================
        if role_type == "BUILD":
            if current_tile is None:
                has_coop = any(
                    isinstance(tile, dict) and tile.get("kind") == "COOP"
                    for row in tiles for tile in row
                )
                has_pasture = any(
                    isinstance(tile, dict) and tile.get("kind") == "PASTURE"
                    for row in tiles for tile in row
                )
                # FIX: cek isi shed
                has_cow_or_sheep = (
                    state.shed.get("COW", 0) > 0 or state.shed.get("SHEEP", 0) > 0
                )
                has_goose = state.shed.get("GOOSE", 0) > 0

                # Prioritas pasture kalau ada cow/sheep
                if has_cow_or_sheep and not has_pasture:
                    return ["BUILD_PASTURE"]
                if has_goose and not has_coop:
                    return ["BUILD_COOP"]
                # FIX: kalau tidak ada animal, JANGAN build
                return None

        # ============================================================
        # PICKUP animal dari shed
        # ============================================================
        if role_type in {"ANIMAL", "PLACE"}:
            if state.is_shed_adjacent(unit_pos):
                # FIX: tambah SHEEP
                if not carried("GOOSE") and state.shed.get("GOOSE", 0) > 0:
                    return ["PICKUP", "GOOSE", 1]
                if not carried("COW") and state.shed.get("COW", 0) > 0:
                    return ["PICKUP", "COW", 1]
                if not carried("SHEEP") and state.shed.get("SHEEP", 0) > 0:
                    return ["PICKUP", "SHEEP", 1]

        # ============================================================
        # PLACE animal ke struktur — FIX: PASTURE untuk cow/sheep
        # ============================================================
        if role_type in {"ANIMAL", "PLACE"}:
            if isinstance(current_tile, dict):
                kind = current_tile.get("kind")
                if not current_tile.get("animal"):
                    if kind == "COOP" and carried("GOOSE"):
                        return ["PLACE", "GOOSE"]
                    # FIX: cow/sheep HARUS ke PASTURE
                    if kind == "PASTURE":
                        if carried("COW"):
                            return ["PLACE", "COW"]
                        if carried("SHEEP"):
                            return ["PLACE", "SHEEP"]

        # ============================================================
        # FEED — FIX: handler baru
        # ============================================================
        if role_type == "FEED":
            if isinstance(current_tile, dict) and current_tile.get("kind") in {"COOP", "PASTURE"}:
                if current_tile.get("animal") and not current_tile.get("fed_today", False):
                    # Butuh wheat di inventory
                    if carried("WHEAT"):
                        return ["FEED"]
                    # Atau ambil wheat dari shed kalau adjacent
                    if state.is_shed_adjacent(unit_pos) and state.shed.get("WHEAT", 0) > 0:
                        return ["PICKUP", "WHEAT", 1]

        # ============================================================
        # COLLECT_FERTILIZER — FIX: handler baru
        # ============================================================
        if role_type == "COLLECT_FERTILIZER":
            if isinstance(current_tile, dict) and current_tile.get("kind") in {"COOP", "PASTURE"}:
                if current_tile.get("animal") and current_tile.get("fertilizer_available", False):
                    return ["COLLECT_FERTILIZER"]

        # ============================================================
        # CARE — FIX: handler baru
        # ============================================================
        if role_type == "CARE":
            if isinstance(current_tile, dict) and current_tile.get("kind") in {"COOP", "PASTURE"}:
                if current_tile.get("animal") and not current_tile.get("cared_today", False):
                    return ["CARE"]

        # ============================================================
        # DROP inventory — FIX: hanya kalau inventory PENUH atau tidak ada task lain
        # ============================================================
        # Cek dulu apakah ada task produktif di tile ini
        has_productive_task = False
        if isinstance(current_tile, dict):
            k = current_tile.get("kind")
            if k == "PLANT":
                if int(current_tile.get("yield_units", 0)) > 0:
                    has_productive_task = True
                elif not current_tile.get("watered_today", False):
                    has_productive_task = True
            elif k == "WEED":
                has_productive_task = True
            elif k in {"COOP", "PASTURE"} and current_tile.get("animal"):
                if int(current_tile.get("yield_units", 0)) > 0:
                    has_productive_task = True
                elif not current_tile.get("fed_today", False):
                    has_productive_task = True

        if (
            len(inventory) > 0
            and state.is_shed_adjacent(unit_pos)
            and not (carried("GOOSE") or carried("COW") or carried("SHEEP") or carried("WHEAT"))
            and not has_productive_task  # FIX: jangan drop kalau masih bisa kerja
        ):
            return ["DROP"]

        # ============================================================
        # DIG
        # ============================================================
        if role_type == "DIG" and isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
            return ["DIG"]

        # ============================================================
        # PLANT
        # ============================================================
        if role_type == "PLANT" and current_tile is None and state.get_quadrant(ux, uy) in state.unlocked_quadrants:
            candidates = []
            planted = state.planted_counts()
            targets = state.planting_targets()
            for crop, count in state.seeds.items():
                spec = CROP_SPECS.get(crop)
                if count > 0 and spec is not None and state.remaining_days() >= spec.first_yield_day:
                    price = state.market_prices.get(spec.product_name, 0.0)
                    expected = price * spec.max_yield - spec.seed_cost
                    target = max(1, min(count, targets.get(crop, 1)))
                    deficit = max(0, target - planted.get(crop, 0))
                    score = expected / max(1, spec.first_yield_day)
                    coverage_deficit = deficit / target
                    candidates.append((coverage_deficit, score, crop))
            if candidates:
                candidates.sort(reverse=True)
                return ["PLANT", candidates[0][2]]

        # ============================================================
        # WATER
        # ============================================================
        if role_type == "WATER" and isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
            if not current_tile.get("watered_today", False):
                return ["WATER"]

        # ============================================================
        # HARVEST — FIX: lebih fleksibel, tetap cek maturity
        # ============================================================
        if role_type == "HARVEST" and isinstance(current_tile, dict):
            if current_tile.get("kind") == "PLANT":
                yield_units = int(current_tile.get("yield_units", 0))
                crop = str(current_tile.get("crop", ""))
                planted_day = int(current_tile.get("planted_day", 0))
                crop_age = state.day - planted_day
                spec = CROP_SPECS.get(crop)

                first_yield = spec.first_yield_day if spec else 2
                if crop_age >= first_yield and yield_units > 0:
                    return ["HARVEST"]
            # Animal harvest
            if current_tile.get("kind") in {"COOP", "PASTURE"}:
                if current_tile.get("animal") and int(current_tile.get("yield_units", 0)) > 0:
                    return ["HARVEST"]

        return None