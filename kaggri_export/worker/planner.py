"""WorkerPlanner orchestrator with anti-stuck protection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from state import FarmState, Pos
from worker.assignment import TaskAssigner
from worker.pathfinding import PathFinder
from worker.roles import WorkerRole
from worker.tasks import TaskEvaluator

class WorkerPlanner:
    def decide_action(
        self,
        unit_pos: Pos,
        state: FarmState,
        inventory: List[Dict[str, Any]],
        role: WorkerRole,
        assigned_targets: Set[Pos],
        other_worker_positions: Optional[Set[Pos]] = None,
    ) -> List[Any]:

        if other_worker_positions is None:
            other_worker_positions = set()

        # 1. Attempt immediate action on current tile for primary task
        primary_task = TaskAssigner.ROLE_HIERARCHY[role][0]
        immediate_act = TaskEvaluator.check_immediate_tile_action(
            unit_pos, state, inventory, primary_task
        )
        if immediate_act:
            return immediate_act

        # 2. Check immediate action for secondary/fallback tasks
        for fallback_task in TaskAssigner.ROLE_HIERARCHY[role][1:]:
            fallback_act = TaskEvaluator.check_immediate_tile_action(
                unit_pos, state, inventory, fallback_task
            )
            if fallback_act:
                return fallback_act

        # 3. Pathfind towards best target
        best_target, target_task = TaskAssigner.find_best_target(
            unit_pos=unit_pos,
            state=state,
            role=role,
            assigned_targets=assigned_targets,
            current_committed_target=None,
            occupied_tiles=other_worker_positions,
        )

        if best_target and target_task:
            # ============================================================
            # ANTI-STUCK: Jika worker sudah di target tapi tidak bisa
            # bertindak (langkah 1 & 2 sudah gagal), JANGAN stuck di sini.
            # Tandai target ini sebagai "tidak bisa dikerjakan sekarang"
            # dan cari target lain.
            # ============================================================
            if best_target == unit_pos:
                # Worker sudah di posisi target tapi tidak bisa bertindak.
                # Ini berarti is_tile_valid_for_task dan check_immediate_tile_action
                # tidak konsisten, ATAU kondisi berubah di tengah turn.
                # Tambahkan ke assigned_targets agar tidak dipilih lagi turn ini.
                assigned_targets.add(best_target)
                # Cari target alternatif (bukan posisi sendiri)
                alt_target, alt_task = TaskAssigner.find_best_target(
                    unit_pos=unit_pos,
                    state=state,
                    role=role,
                    assigned_targets=assigned_targets,
                    current_committed_target=None,
                    occupied_tiles=other_worker_positions,
                )
                if alt_target and alt_task:
                    assigned_targets.add(alt_target)
                    move_cmd = PathFinder.a_star_next_step(
                        unit_pos, alt_target, occupied_tiles=other_worker_positions
                    )
                    if move_cmd != "PASS":
                        return [move_cmd]
                return ["PASS"]

            assigned_targets.add(best_target)
            move_cmd = PathFinder.a_star_next_step(
                unit_pos, best_target, occupied_tiles=other_worker_positions
            )
            if move_cmd != "PASS":
                return [move_cmd]

            assigned_targets.add(best_target)
            alt_target, alt_task = TaskAssigner.find_best_target(
                unit_pos=unit_pos,
                state=state,
                role=role,
                assigned_targets=assigned_targets,
                current_committed_target=None,
                occupied_tiles=other_worker_positions,
            )
            if alt_target and alt_task:
                assigned_targets.add(alt_target)
                alt_move = PathFinder.a_star_next_step(
                    unit_pos, alt_target, occupied_tiles=other_worker_positions
                )
                if alt_move != "PASS":
                    return [alt_move]

        return ["PASS"]