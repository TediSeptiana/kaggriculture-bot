"""A* pathfinding algorithm for field unit navigation with collision avoidance."""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple
from state import Pos

class PathFinder:
    """Calculates spatial Manhattan metrics and A* navigation steps."""

    BOARD_SIZE: int = 10

    DIRECTIONS: Dict[str, Tuple[int, int]] = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0),
    }

    @staticmethod
    def manhattan_distance(pos1: Pos, pos2: Pos) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @classmethod
    def a_star_next_step(
        cls, 
        start: Pos, 
        target: Pos, 
        occupied_tiles: Optional[Set[Pos]] = None
    ) -> str:
        """Finds the optimal first step towards target, avoiding occupied tiles."""
        if start == target:
            return "PASS"

        if occupied_tiles is None:
            occupied_tiles = set()
        else:
            # Pastikan posisi diri sendiri tidak dianggap sebagai hambatan
            occupied_tiles = occupied_tiles - {start}

        # Pelacakan g_score global untuk mencegah ekspansi node berlebihan
        g_scores: Dict[Pos, int] = {start: 0}
        # Heap: (f_score, g_score, current_pos, first_move)
        open_set: List[Tuple[int, int, Pos, str]] = []

        for direction, (dx, dy) in cls.DIRECTIONS.items():
            neighbor = (start[0] + dx, start[1] + dy)
            if 0 <= neighbor[0] < cls.BOARD_SIZE and 0 <= neighbor[1] < cls.BOARD_SIZE:
                if neighbor in occupied_tiles:
                    continue
                g_score = 1
                f_score = g_score + cls.manhattan_distance(neighbor, target)
                heapq.heappush(open_set, (f_score, g_score, neighbor, direction))
                g_scores[neighbor] = g_score

        visited: Set[Pos] = set()

        while open_set:
            f_score, g_score, current, first_move = heapq.heappop(open_set)

            if current == target:
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
                    and neighbor not in occupied_tiles
                ):
                    tentative_g = g_score + 1
                    # Hanya proses jika ini jalur yang lebih pendek ke node tersebut
                    if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                        g_scores[neighbor] = tentative_g
                        h_score = cls.manhattan_distance(neighbor, target)
                        next_f = tentative_g + h_score
                        heapq.heappush(open_set, (next_f, tentative_g, neighbor, first_move))

        return "PASS"