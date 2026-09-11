"""Worker and unit decision-making logic for Farmers and Farm Hands."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
from kaggriculture.state import CROP_SPECS, FarmState, Pos, Tile


class WorkerPlanner:
    """Calculates tactical turn actions for individual field workers."""

    @staticmethod
    def manhattan_distance(pos1: Pos, pos2: Pos) -> int:
        """Calculates Manhattan distance between two grid points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @classmethod
    def get_step_towards(cls, current: Pos, target: Pos) -> str:
        """Generates direction command towards target coordinate."""
        cx, cy = current
        tx, ty = target
        if cx < tx:
            return "EAST"
        if cx > tx:
            return "WEST"
        if cy < ty:
            return "SOUTH"
        if cy > ty:
            return "NORTH"
        return "PASS"

    def decide_action(
        self,
        unit_pos: Pos,
        state: FarmState,
        inventory: List[Dict[str, Any]],
    ) -> List[Any]:
        """Evaluates tile condition and returns high-priority action sequence."""
        ux, uy = unit_pos
        tiles = state.tiles
        current_tile: Tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
        unlocked_positions = state.get_unlocked_tiles()

        # Priority 1: Clear inventory at shed if holding items
        if len(inventory) > 0 and state.is_shed_adjacent(unit_pos):
            return ["DROP"]

        # Priority 2: Clear Weed
        if isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
            return ["DIG"]

        # Priority 3: Harvest mature crops
        if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
            yield_units = int(current_tile.get("yield_units", 0))
            crop = str(current_tile.get("crop", ""))
            planted_day = int(current_tile.get("planted_day", 0))
            crop_age = state.day - planted_day
            spec = CROP_SPECS.get(crop)

            first_yield = spec.first_yield_day if spec else 2
            if crop_age >= first_yield and yield_units > 0:
                return ["HARVEST"]

            if not current_tile.get("watered_today", False):
                return ["WATER"]

        # Priority 4: Plant seeds on empty unlocked tile
        if current_tile is None and state.get_quadrant(ux, uy) in state.unlocked_quadrants:
            if state.seeds.get("WHEAT", 0) > 0:
                return ["PLANT", "WHEAT"]
            if state.seeds.get("CARROT", 0) > 0:
                return ["PLANT", "CARROT"]
            if state.seeds.get("MELON", 0) > 0:
                return ["PLANT", "MELON"]

        # Priority 5: Pathfinding towards priority task
        best_target: Optional[Pos] = None
        min_dist = 999

        for pos in unlocked_positions:
            tx, ty = pos
            t = tiles[ty][tx]

            needs_action = False
            if t is None and sum(state.seeds.values()) > 0:
                needs_action = True
            elif isinstance(t, dict):
                if t.get("kind") == "WEED":
                    needs_action = True
                elif t.get("kind") == "PLANT" and not t.get("watered_today", False):
                    needs_action = True

            if needs_action:
                d = self.manhattan_distance(unit_pos, pos)
                if d < min_dist:
                    min_dist = d
                    best_target = pos

        if best_target:
            move_cmd = self.get_step_towards(unit_pos, best_target)
            if move_cmd != "PASS":
                return [move_cmd]

        return ["PASS"]