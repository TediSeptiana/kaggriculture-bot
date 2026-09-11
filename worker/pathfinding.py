"""A* pathfinding algorithm for field unit navigation."""

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
        """Calculates Manhattan distance heuristic h(n)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @classmethod
    def a_star_next_step(cls, start: Pos, target: Pos) -> str:
        """Finds the optimal first step towards target position using A*."""
        if start == target:
            return "PASS"

        open_set: List[Tuple[int, int, Pos, Optional[str]]] = []

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