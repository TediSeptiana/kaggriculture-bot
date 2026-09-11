"""Target selection and fallback task assignment solver."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from state import CROP_SPECS, FarmState, Pos, Tile
from worker.pathfinding import PathFinder
from worker.roles import WorkerRole


class TaskAssigner:
    """Finds nearest task targets with primary role priority and dynamic fallback."""

    # Role execution hierarchy: Primary task -> Secondary Fallback task
    ROLE_HIERARCHY: Dict[WorkerRole, List[str]] = {
        WorkerRole.DIGGER: ["DIG", "WATER", "HARVEST", "PLANT"],    # Farmer falls back to WATER/HARVEST/PLANT
        WorkerRole.PLANTER: ["PLANT", "WATER", "HARVEST", "DIG"],
        WorkerRole.WATERER: ["WATER", "HARVEST", "DIG", "PLANT"],
        WorkerRole.HARVESTER: ["HARVEST", "WATER", "DIG", "PLANT"],
        WorkerRole.VERSATILE: ["HARVEST", "WATER", "PLANT", "DIG"],
    }

    @classmethod
    def is_tile_valid_for_task(
        cls, pos: Pos, state: FarmState, task_type: str
    ) -> bool:
        """Determines if a tile requires a specific task."""
        tx, ty = pos
        t = state.tiles[ty][tx]

        if task_type == "DIG":
            return isinstance(t, dict) and t.get("kind") == "WEED"

        if task_type == "PLANT":
            return t is None and sum(state.seeds.values()) > 0

        if task_type == "WATER":
            return isinstance(t, dict) and t.get("kind") == "PLANT" and not t.get("watered_today", False)

        if task_type == "HARVEST":
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                yield_units = int(t.get("yield_units", 0))
                crop = str(t.get("crop", ""))
                planted_day = int(t.get("planted_day", 0))
                crop_age = state.day - planted_day
                spec = CROP_SPECS.get(crop)
                first_yield = spec.first_yield_day if spec else 2
                return crop_age >= first_yield and yield_units > 0

        return False

    @classmethod
    def find_best_target(
        cls,
        unit_pos: Pos,
        state: FarmState,
        role: WorkerRole,
        assigned_targets: Set[Pos],
    ) -> Tuple[Optional[Pos], Optional[str]]:
        """Finds nearest valid target position based on primary role and fallback hierarchy."""
        unlocked_positions = state.get_unlocked_tiles()
        task_order = cls.ROLE_HIERARCHY.get(role, ["WATER", "HARVEST", "PLANT", "DIG"])

        for task_type in task_order:
            best_target: Optional[Pos] = None
            min_dist = 999

            for pos in unlocked_positions:
                if pos in assigned_targets:
                    continue

                if cls.is_tile_valid_for_task(pos, state, task_type):
                    d = PathFinder.manhattan_distance(unit_pos, pos)
                    if d < min_dist:
                        min_dist = d
                        best_target = pos

            if best_target is not None:
                return best_target, task_type

        return None, None