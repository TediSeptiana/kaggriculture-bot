"""Role-based Worker and unit decision-making logic using A* pathfinding."""

from __future__ import annotations

import heapq
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
from state import CROP_SPECS, FarmState, Pos, Tile


class WorkerRole(Enum):
    """Worker specialization roles."""

    DIGGER = auto()      # Main Farmer: Focuses on DIGWEEDS / DIG
    PLANTER = auto()     # Hand 0: Focuses on PLANTING seeds
    WATERER = auto()     # Hand 1: Focuses on WATERING unwatered crops
    HARVESTER = auto()   # Hand 2: Focuses on HARVESTING mature crops


class WorkerPlanner:
    """Calculates tactical turn actions using role isolation and A* pathfinding."""

    BOARD_SIZE: int = 10

    DIRECTIONS: Dict[str, Tuple[int, int]] = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0),
    }

    @staticmethod
    def manhattan_distance(pos1: Pos, pos2: Pos) -> int:
        """Calculates Manhattan distance heuristic h(n)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @classmethod
    def a_star_next_step(cls, start: Pos, target: Pos) -> str:
        """Finds the optimal first step towards a target position using A*."""
        if start == target:
            return "PASS"

        open_set: List[Tuple[int, int, Pos, Optional[str]]] = []
        h_start = cls.manhattan_distance(start, target)

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
        role: WorkerRole,
        assigned_targets: Set[Pos],
    ) -> List[Any]:
        """Evaluates tile conditions and returns specialized actions based on strict WorkerRole."""
        ux, uy = unit_pos
        tiles = state.tiles
        current_tile: Tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
        unlocked_positions = state.get_unlocked_tiles()

        # Universal Logistics Priority: Drop items at shed if carrying produce/items
        if len(inventory) > 0 and state.is_shed_adjacent(unit_pos):
            return ["DROP"]

        # --- ROLE-SPECIFIC TILE ACTION EXECUTION ---

        # 1. DIGGER (Farmer): Clear weeds
        if role == WorkerRole.DIGGER:
            if isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
                return ["DIG"]

        # 2. PLANTER (Hand 0): Plant seeds on empty unlocked tile
        elif role == WorkerRole.PLANTER:
            if current_tile is None and state.get_quadrant(ux, uy) in state.unlocked_quadrants:
                if state.seeds.get("WHEAT", 0) > 0:
                    return ["PLANT", "WHEAT"]
                if state.seeds.get("CARROT", 0) > 0:
                    return ["PLANT", "CARROT"]
                if state.seeds.get("MELON", 0) > 0:
                    return ["PLANT", "MELON"]

        # 3. WATERER (Hand 1): Water unwatered plants
        elif role == WorkerRole.WATERER:
            if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
                if not current_tile.get("watered_today", False):
                    return ["WATER"]

        # 4. HARVESTER (Hand 2): Harvest mature crops
        elif role == WorkerRole.HARVESTER:
            if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
                yield_units = int(current_tile.get("yield_units", 0))
                crop = str(current_tile.get("crop", ""))
                planted_day = int(current_tile.get("planted_day", 0))
                crop_age = state.day - planted_day
                spec = CROP_SPECS.get(crop)

                first_yield = spec.first_yield_day if spec else 2
                if crop_age >= first_yield and yield_units > 0:
                    return ["HARVEST"]

        # --- A* PATHFINDING TOWARDS ROLE-SPECIFIC TARGETS ---
        best_target: Optional[Pos] = None
        min_dist = 999

        for pos in unlocked_positions:
            if pos in assigned_targets:
                continue  # Hindari bentrokan target dengan unit lain

            tx, ty = pos
            t = tiles[ty][tx]

            target_valid = False

            if role == WorkerRole.DIGGER:
                if isinstance(t, dict) and t.get("kind") == "WEED":
                    target_valid = True

            elif role == WorkerRole.PLANTER:
                if t is None and sum(state.seeds.values()) > 0:
                    target_valid = True

            elif role == WorkerRole.WATERER:
                if isinstance(t, dict) and t.get("kind") == "PLANT" and not t.get("watered_today", False):
                    target_valid = True

            elif role == WorkerRole.HARVESTER:
                if isinstance(t, dict) and t.get("kind") == "PLANT":
                    yield_units = int(t.get("yield_units", 0))
                    crop = str(t.get("crop", ""))
                    planted_day = int(t.get("planted_day", 0))
                    crop_age = state.day - planted_day
                    spec = CROP_SPECS.get(crop)
                    first_yield = spec.first_yield_day if spec else 2
                    if crop_age >= first_yield and yield_units > 0:
                        target_valid = True

            if target_valid:
                d = self.manhattan_distance(unit_pos, pos)
                if d < min_dist:
                    min_dist = d
                    best_target = pos

        if best_target:
            assigned_targets.add(best_target)
            move_cmd = self.a_star_next_step(unit_pos, best_target)
            if move_cmd != "PASS":
                return [move_cmd]

        return ["PASS"]