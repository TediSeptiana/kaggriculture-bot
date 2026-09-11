"""Worker and unit decision-making logic for Farmers and Farm Hands using A* pathfinding."""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Set, Tuple
from state import CROP_SPECS, FarmState, Pos, Tile


class WorkerPlanner:
    """Calculates tactical turn actions and optimal pathfinding for individual field workers."""

    # Grid boundary constants (Kaggriculture board size is 10x10)
    BOARD_SIZE: int = 10

    # Orthogonal movement mapping
    DIRECTIONS: Dict[str, Tuple[int, int]] = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0),
    }

    @staticmethod
    def manhattan_distance(pos1: Pos, pos2: Pos) -> int:
        """Calculates Manhattan distance heuristic function h(n) between two points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @classmethod
    def a_star_next_step(cls, start: Pos, target: Pos) -> str:
        """Finds the optimal first step towards a target position using A* algorithm.

        Args:
            start: Current worker position (x, y).
            target: Target tile position (x, y).

        Returns:
            The directional action string ("NORTH", "SOUTH", "EAST", "WEST", or "PASS").
        """
        if start == target:
            return "PASS"

        # Priority queue stores tuples: (f_score, g_score, current_pos, first_move)
        open_set: List[Tuple[int, int, Pos, Optional[str]]] = []
        
        # Initial evaluation from start
        h_start = cls.manhattan_distance(start, target)
        
        # Populate initial moves
        for direction, (dx, dy) in cls.DIRECTIONS.items():
            neighbor = (start[0] + dx, start[1] + dy)
            if 0 <= neighbor[0] < cls.BOARD_SIZE and 0 <= neighbor[1] < cls.BOARD_SIZE:
                g_score = 1
                f_score = g_score + cls.manhattan_distance(neighbor, target)
                heapq.heappush(open_set, (f_score, g_score, neighbor, direction))

        visited: Set[Pos] = {start}

        while open_set:
            f_score, g_score, current, first_move = heapq.heappop(open_set)

            if current == target and first_move is not None:
                return first_move

            if current in visited:
                continue
            visited.add(current)

            for direction, (dx, dy) in cls.DIRECTIONS.items():
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Bounds check (0..9)
                if (
                    0 <= neighbor[0] < cls.BOARD_SIZE
                    and 0 <= neighbor[1] < cls.BOARD_SIZE
                    and neighbor not in visited
                ):
                    tentative_g = g_score + 1
                    h_score = cls.manhattan_distance(neighbor, target)
                    next_f = tentative_g + h_score
                    heapq.heappush(open_set, (next_f, tentative_g, neighbor, first_move))

        return "PASS"

    def decide_action(
        self,
        unit_pos: Pos,
        state: FarmState,
        inventory: List[Dict[str, Any]],
    ) -> List[Any]:
        """Evaluates tile condition and returns high-priority action sequence using A*."""
        ux, uy = unit_pos
        tiles = state.tiles
        current_tile: Tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
        unlocked_positions = state.get_unlocked_tiles()

        # Priority 1: Clear inventory at shed if holding items
        if len(inventory) > 0 and state.is_shed_adjacent(unit_pos):
            return ["DROP"]

        # Priority 2: Clear Weed on current tile
        if isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
            return ["DIG"]

        # Priority 3: Harvest mature crops on current tile
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

        # Priority 5: Pathfinding towards priority task using A* Search
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
            move_cmd = self.a_star_next_step(unit_pos, best_target)
            if move_cmd != "PASS":
                return [move_cmd]

        return ["PASS"]