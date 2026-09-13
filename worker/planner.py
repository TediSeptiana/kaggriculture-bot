"""WorkerPlanner orchestrator maintaining backward compatibility."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from state import FarmState, Pos
from worker.assignment import TaskAssigner
from worker.pathfinding import PathFinder
from worker.roles import WorkerRole
from worker.tasks import TaskEvaluator


class WorkerPlanner:
    """Calculates tactical turn actions using role isolation and dynamic fallback execution."""

    def decide_action(
        self,
        unit_pos: Pos,
        state: FarmState,
        inventory: List[Dict[str, Any]],
        role: WorkerRole,
        assigned_targets: Set[Pos],
    ) -> List[Any]:
        """Evaluates tile conditions and executes task actions with fallback support."""
        # 1. Attempt immediate action on current tile for primary task
        primary_task = TaskAssigner.ROLE_HIERARCHY[role][0]
        immediate_act = TaskEvaluator.check_immediate_tile_action(unit_pos, state, inventory, primary_task)
        if immediate_act:
            return immediate_act

        # 2. Check immediate action for secondary/fallback tasks if primary is unavailable
        for fallback_task in TaskAssigner.ROLE_HIERARCHY[role][1:]:
            fallback_act = TaskEvaluator.check_immediate_tile_action(unit_pos, state, inventory, fallback_task)
            if fallback_act:
                return fallback_act

        # 3. Pathfind towards best target (Primary or Fallback)
        best_target, target_task = TaskAssigner.find_best_target(unit_pos, state, role, assigned_targets)

        if best_target and target_task:
            assigned_targets.add(best_target)
            move_cmd = PathFinder.a_star_next_step(unit_pos, best_target)
            if move_cmd != "PASS":
                return [move_cmd]

        return ["PASS"]