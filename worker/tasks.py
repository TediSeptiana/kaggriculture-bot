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
        """Checks if unit can perform immediate action on current tile."""
        ux, uy = unit_pos
        tiles = state.tiles
        current_tile: Tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None

        # Universal Priority: Drop produce at shed if carrying inventory
        if len(inventory) > 0 and state.is_shed_adjacent(unit_pos):
            return ["DROP"]

        if role_type == "DIG" and isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
            return ["DIG"]

        if role_type == "PLANT" and current_tile is None and state.get_quadrant(ux, uy) in state.unlocked_quadrants:
            for crop in ("WHEAT",):
                if state.seeds.get(crop, 0) > 0:
                    return ["PLANT", crop]

        if role_type == "WATER" and isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
            if not current_tile.get("watered_today", False):
                return ["WATER"]

        if role_type == "HARVEST" and isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
            yield_units = int(current_tile.get("yield_units", 0))
            crop = str(current_tile.get("crop", ""))
            planted_day = int(current_tile.get("planted_day", 0))
            crop_age = state.day - planted_day
            spec = CROP_SPECS.get(crop)

            first_yield = spec.first_yield_day if spec else 2
            if crop_age >= first_yield and yield_units > 0:
                return ["HARVEST"]

        return None