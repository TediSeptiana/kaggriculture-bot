"""Target selection, persistence, and fallback task assignment solver."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from state import CROP_SPECS, FarmState, Pos, Tile
from worker.pathfinding import PathFinder
from worker.roles import WorkerRole


class TaskAssigner:
    """Finds nearest task targets with sticky commitments, primary role priority, and dynamic fallback."""

    # Dynamic fallback hierarchy per role
    ROLE_HIERARCHY: Dict[WorkerRole, List[str]] = {
        WorkerRole.DIGGER: ["DIG", "WATER", "HARVEST", "PLANT"],
        WorkerRole.PLANTER: ["PLANT", "WATER", "HARVEST", "DIG"],
        WorkerRole.WATERER: ["WATER", "HARVEST", "DIG", "PLANT"],
        WorkerRole.HARVESTER: ["HARVEST", "WATER", "DIG", "PLANT"],
        WorkerRole.VERSATILE: ["HARVEST", "WATER", "PLANT", "DIG"],
    }

    @classmethod
    def is_tile_valid_for_task(
        cls, pos: Pos, state: FarmState, task_type: str
    ) -> bool:
        """Determines if a tile strictly requires the specified task."""
        tx, ty = pos
        tiles = state.tiles
        if ty >= len(tiles) or tx >= len(tiles[ty]):
            return False

        t = tiles[ty][tx]

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
    def is_target_still_valid(
        cls, pos: Pos, state: FarmState, role: WorkerRole
    ) -> bool:
        """Verifies if the unit's locked target position remains valid for any of its executable tasks."""
        task_order = cls.ROLE_HIERARCHY.get(role, ["WATER", "HARVEST", "PLANT", "DIG"])
        for task_type in task_order:
            if cls.is_tile_valid_for_task(pos, state, task_type):
                return True
        return False

    @classmethod
    def find_best_target(
        cls,
        unit_pos: Pos,
        state: FarmState,
        role: WorkerRole,
        assigned_targets: Set[Pos],
        current_committed_target: Optional[Pos] = None,
    ) -> Tuple[Optional[Pos], Optional[str]]:
        """Finds nearest valid target position incorporating sticky hysteresis to avoid thrashing."""
        # 1. Evaluate current committed target if valid and unassigned to another worker
        if current_committed_target is not None:
            if (
                current_committed_target not in assigned_targets
                and cls.is_target_still_valid(current_committed_target, state, role)
            ):
                # Identify which task type applies to this committed position
                task_order = cls.ROLE_HIERARCHY.get(role, ["WATER", "HARVEST", "PLANT", "DIG"])
                for task_type in task_order:
                    if cls.is_tile_valid_for_task(current_committed_target, state, task_type):
                        return current_committed_target, task_type

        # 2. Find new optimal target across hierarchy if no valid committed target exists
        unlocked_positions = state.get_unlocked_tiles()
        task_order = cls.ROLE_HIERARCHY.get(role, ["WATER", "HARVEST", "PLANT", "DIG"])

        for task_type in task_order:
            best_target: Optional[Pos] = None
            min_dist = 9999

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