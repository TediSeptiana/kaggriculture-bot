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
                has_cow_or_sheep = (
                    state.shed.get("COW", 0) > 0 or state.shed.get("SHEEP", 0) > 0
                )
                has_goose = state.shed.get("GOOSE", 0) > 0

                if has_cow_or_sheep and not has_pasture:
                    return ["BUILD_PASTURE"]
                if has_goose and not has_coop:
                    return ["BUILD_COOP"]
                return None

        # ============================================================
        # PICKUP animal dari shed
        # ============================================================
        if role_type in {"ANIMAL", "PLACE"}:
            if state.is_shed_adjacent(unit_pos):
                if not carried("GOOSE") and state.shed.get("GOOSE", 0) > 0:
                    return ["PICKUP", "GOOSE", 1]
                if not carried("COW") and state.shed.get("COW", 0) > 0:
                    return ["PICKUP", "COW", 1]
                if not carried("SHEEP") and state.shed.get("SHEEP", 0) > 0:
                    return ["PICKUP", "SHEEP", 1]

        # ============================================================
        # ============================================================
        # ANIMAL — urutan prioritas:
        #   1. Preemptive pickup WHEAT dari shed (kalau ada hewan lapar)
        #   2. COLLECT egg/milk/wool (paling urgent — yield bisa terbuang)
        #   3. COLLECT_FERTILIZER (nilai $100+, tidak accumulate)
        #   4. FEED (wajib tiap hari, kalau tidak hewan kabur)
        # ============================================================
                # ============================================================
        # ANIMAL — prioritas FEED dulu (animal bisa mati kalau tidak di-feed)
        # ============================================================
        if role_type == "ANIMAL":
            has_hungry_animal = any(
                isinstance(t, dict)
                and t.get("kind") in ("COOP", "PASTURE")
                and t.get("animal")
                and not t.get("fed_today", False)
                for row in tiles for t in row
            )

            # Preemptive pickup wheat dari shed
            if (
                has_hungry_animal
                and not carried("WHEAT")
                and state.is_shed_adjacent(unit_pos)
                and state.shed.get("WHEAT", 0) > 0
            ):
                return ["PICKUP", "WHEAT", 5]

            if isinstance(current_tile, dict) and current_tile.get("kind") in ("COOP", "PASTURE"):
                animal = current_tile.get("animal")
                if animal:
                    # PRIORITAS 1: FEED — animal mati kalau tidak di-feed
                    if not current_tile.get("fed_today", False) and carried("WHEAT"):
                        return ["FEED"]
                    # PRIORITAS 2: COLLECT egg/milk/wool
                    if int(current_tile.get("yield_units", 0)) > 0:
                        return ["COLLECT"]
                    # PRIORITAS 3: COLLECT fertilizer
                    if current_tile.get("fertilizer_available", False):
                        return ["COLLECT_FERTILIZER"]
        # PLACE animal ke struktur
        # ============================================================
        if role_type in {"ANIMAL", "PLACE"}:
            if isinstance(current_tile, dict):
                kind = current_tile.get("kind")
                if not current_tile.get("animal"):
                    if kind == "COOP" and carried("GOOSE"):
                        return ["PLACE", "GOOSE"]
                    if kind == "PASTURE":
                        if carried("COW"):
                            return ["PLACE", "COW"]
                        if carried("SHEEP"):
                            return ["PLACE", "SHEEP"]

        # ============================================================
        # FERTILIZE — apply fertilizer ke tanaman (MELON + STRAWBERRY)
        # ============================================================
        if role_type == "FERTILIZE":
            # 1. Pickup fertilizer dari shed jika belum bawa
            if state.is_shed_adjacent(unit_pos) and not carried("FERTILIZER"):
                if state.shed.get("FERTILIZER", 0) > 0:
                    return ["PICKUP", "FERTILIZER", 1]

            # 2. Apply ke tile PLANT di bonus window
            if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
                if carried("FERTILIZER"):
                    crop = str(current_tile.get("crop", ""))
                    planted_day = int(current_tile.get("planted_day", 0))
                    age = state.day - planted_day
                    fert_until = int(current_tile.get("fertilized_until_day", -1))
                    spec = CROP_SPECS.get(crop)

                    if spec is not None and fert_until < state.day:
                        # Bonus window: mulai dari ceil(max_yield_day / 2)
                        bonus_start = (spec.max_yield_day + 1) // 2
                        if bonus_start <= age <= spec.max_yield_day:
                            return ["FERTILIZE"]

        # ============================================================
        # FEED — untuk role khusus FEED (fallback ke ANIMAL di atas)
        # ============================================================
        if role_type == "FEED":
            has_hungry_animal = any(
                isinstance(t, dict)
                and t.get("kind") in ("COOP", "PASTURE")
                and t.get("animal")
                and not t.get("fed_today", False)
                for row in tiles for t in row
            )

            if (
                has_hungry_animal
                and not carried("WHEAT")
                and state.is_shed_adjacent(unit_pos)
                and state.shed.get("WHEAT", 0) > 0
            ):
                return ["PICKUP", "WHEAT", 5]

            if isinstance(current_tile, dict) and current_tile.get("kind") in {"COOP", "PASTURE"}:
                if current_tile.get("animal") and not current_tile.get("fed_today", False):
                    if carried("WHEAT"):
                        return ["FEED"]

        # ============================================================
        # COLLECT_FERTILIZER — role khusus
        # ============================================================
        if role_type == "COLLECT_FERTILIZER":
            if isinstance(current_tile, dict) and current_tile.get("kind") in {"COOP", "PASTURE"}:
                if current_tile.get("animal") and current_tile.get("fertilizer_available", False):
                    return ["COLLECT_FERTILIZER"]

        # ============================================================
        # CARE
        # ============================================================
        if role_type == "CARE":
            if isinstance(current_tile, dict) and current_tile.get("kind") in {"COOP", "PASTURE"}:
                if current_tile.get("animal") and not current_tile.get("cared_today", False):
                    return ["CARE"]

        # ============================================================
        # DROP inventory
        # ============================================================
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
            and not has_productive_task
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
        # HARVEST
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
            if current_tile.get("kind") in {"COOP", "PASTURE"}:
                if current_tile.get("animal") and int(current_tile.get("yield_units", 0)) > 0:
                    return ["HARVEST"]

        return None