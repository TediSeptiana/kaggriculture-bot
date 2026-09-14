"""Target selection with spatial zoning and collision avoidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from state import CROP_SPECS, FarmState, Pos, Tile
from worker.pathfinding import PathFinder
from worker.roles import WorkerRole

class Zone:
    SHED = "SHED"
    FARM = "FARM"
    ANIMAL = "ANIMAL"
    ANY = "ANY"

class TaskAssigner:
    ROLE_HIERARCHY: Dict[WorkerRole, List[str]] = {
        WorkerRole.DIGGER: ["DIG", "HARVEST", "WATER", "PLANT"],
        WorkerRole.PLANTER: ["PLANT", "WATER", "HARVEST", "DIG"],
        WorkerRole.WATERER: ["WATER", "HARVEST", "DIG", "PLANT"],
        WorkerRole.HARVESTER: ["HARVEST", "WATER", "PLANT", "DIG"],
        WorkerRole.ANIMAL: ["COLLECT_FERTILIZER", "FEED", "PLACE", "BUILD", "HARVEST", "WATER", "PLANT", "DIG"],
        WorkerRole.VERSATILE: ["HARVEST", "WATER", "PLANT", "DIG"],
        WorkerRole.FERTILIZE: ["FERTILIZE", "HARVEST", "WATER", "PLANT", "DIG"],
    }

    ROLE_PREFERRED_ZONES: Dict[WorkerRole, str] = {
        WorkerRole.DIGGER: Zone.FARM,
        WorkerRole.PLANTER: Zone.FARM,
        WorkerRole.WATERER: Zone.FARM,
        WorkerRole.HARVESTER: Zone.FARM,
        WorkerRole.ANIMAL: Zone.ANIMAL,
        WorkerRole.VERSATILE: Zone.ANY,
        WorkerRole.FERTILIZE: Zone.FARM,
    }

    CROP_PRIORITY: List[str] = ["MELON", "CARROT", "WHEAT", "TOMATO", "STRAWBERRY"]

    @classmethod
    def get_tile_zone(cls, pos: Pos, state: FarmState) -> str:
        if state.is_shed_adjacent(pos):
            return Zone.SHED
        tiles = state.tiles
        tx, ty = pos
        if ty < len(tiles) and tx < len(tiles[ty]):
            t = tiles[ty][tx]
            if isinstance(t, dict) and t.get("kind") in {"COOP", "PASTURE"}:
                return Zone.ANIMAL
        return Zone.FARM

    @classmethod
    def is_tile_valid_for_task(cls, pos: Pos, state: FarmState, task_type: str) -> bool:
        """
        HARUS KONSISTEN dengan TaskEvaluator.check_immediate_tile_action di tasks.py.
        Jika fungsi ini return True, maka check_immediate_tile_action JUGA harus return action.
        """
        tx, ty = pos
        tiles = state.tiles
        if ty >= len(tiles) or tx >= len(tiles[ty]):
            return False
        t = tiles[ty][tx]

        if task_type == "DIG":
            return isinstance(t, dict) and t.get("kind") == "WEED"

        if task_type == "PLANT":
            if t is not None:
                return False
            return cls.choose_plant_crop(state) is not None

        if task_type == "WATER":
            return (
                isinstance(t, dict)
                and t.get("kind") == "PLANT"
                and not t.get("watered_today", False)
            )

        if task_type == "HARVEST":
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                yield_units = int(t.get("yield_units", 0))
                crop = str(t.get("crop", ""))
                planted_day = int(t.get("planted_day", 0))
                spec = CROP_SPECS.get(crop)
                first_yield = spec.first_yield_day if spec else 2
                return (state.day - planted_day) >= first_yield and yield_units > 0
            if isinstance(t, dict) and t.get("kind") in {"COOP", "PASTURE"}:
                if int(t.get("yield_units", 0)) > 0:
                    return True
                if t.get("animal") and t.get("fertilizer_available", False):
                    return True
            return False

        if task_type == "FERTILIZE":
            if not isinstance(t, dict) or t.get("kind") != "PLANT":
                return False
            crop = str(t.get("crop", ""))
            if crop not in ("MELON", "STRAWBERRY", "TOMATO"):
                return False
            spec = CROP_SPECS.get(crop)
            if spec is None:
                return False
            planted_day = int(t.get("planted_day", 0))
            age = state.day - planted_day
            fert_until = int(t.get("fertilized_until_day", -1))
            if fert_until >= state.day:
                return False
            bonus_start = (spec.max_yield_day + 1) // 2
            return bonus_start <= age <= spec.max_yield_day

        # ============================================================
        # PERBAIKAN KRITIS: Semua task animal harus cek kondisi LENGKAP
        # agar konsisten dengan check_immediate_tile_action di tasks.py
        # ============================================================
        if task_type == "COLLECT_FERTILIZER":
            if not isinstance(t, dict) or t.get("kind") not in {"COOP", "PASTURE"}:
                return False
            if not t.get("animal"):
                return False
            # WAJIB cek fertilizer_available! Ini yang menyebabkan stuck!
            return bool(t.get("fertilizer_available", False))

        if task_type == "FEED":
            if not isinstance(t, dict) or t.get("kind") not in {"COOP", "PASTURE"}:
                return False
            if not t.get("animal"):
                return False
            # WAJIB cek fed_today! Jangan feed animal yang sudah kenyang
            return not t.get("fed_today", False)

        if task_type == "PLACE":
            if not isinstance(t, dict) or t.get("kind") not in {"COOP", "PASTURE"}:
                return False
            # Hanya valid jika struktur kosong (belum ada animal)
            return not t.get("animal")

        if task_type == "BUILD":
            # BUILD valid di tile kosong jika belum ada struktur yang dibutuhkan
            if t is not None:
                return False
            has_coop = any(
                isinstance(tile, dict) and tile.get("kind") == "COOP"
                for row in tiles for tile in row
            )
            has_pasture = any(
                isinstance(tile, dict) and tile.get("kind") == "PASTURE"
                for row in tiles for tile in row
            )
            needs_coop = not has_coop and state.shed.get("GOOSE", 0) > 0
            needs_pasture = not has_pasture and (
                state.shed.get("COW", 0) > 0 or state.shed.get("SHEEP", 0) > 0
            )
            return needs_coop or needs_pasture

        return False

    @classmethod
    def choose_plant_crop(cls, state: FarmState) -> Optional[str]:
        for crop in cls.CROP_PRIORITY:
            if state.seeds.get(crop, 0) > 0:
                return crop
        for crop, count in state.seeds.items():
            if count > 0:
                return crop
        return None

    @classmethod
    def is_target_still_valid(
        cls, pos: Pos, state: FarmState, role: WorkerRole
    ) -> bool:
        task_order = cls.ROLE_HIERARCHY.get(role, ["HARVEST", "WATER", "PLANT", "DIG"])
        for task_type in task_order:
            if cls.is_tile_valid_for_task(pos, state, task_type):
                return True
        return False

    # ... (bagian atas file tetap sama: Zone, TaskAssigner, get_tile_zone, choose_plant_crop) ...

    @classmethod
    def find_best_target(
        cls,
        unit_pos: Pos,
        state: FarmState,
        role: WorkerRole,
        assigned_targets: Set[Pos],
        current_committed_target: Optional[Pos] = None,
        occupied_tiles: Optional[Set[Pos]] = None,
    ) -> Tuple[Optional[Pos], Optional[str]]:

        if occupied_tiles is None:
            occupied_tiles = set()

        # 1. Cek committed target (jika ada)
        if current_committed_target is not None:
            is_blocked = current_committed_target in (occupied_tiles - {unit_pos})
            if current_committed_target not in assigned_targets and not is_blocked:
                if cls.is_target_still_valid(current_committed_target, state, role):
                    if cls.is_tile_valid_for_task(current_committed_target, state, "HARVEST"):
                        return current_committed_target, "HARVEST"
                    for task_type in cls.ROLE_HIERARCHY.get(role, ["HARVEST", "WATER", "PLANT", "DIG"]):
                        if cls.is_tile_valid_for_task(current_committed_target, state, task_type):
                            return current_committed_target, task_type

        # 2. Cari target baru
        unlocked_positions = state.get_unlocked_tiles()
        task_order = cls.ROLE_HIERARCHY.get(role, ["HARVEST", "WATER", "PLANT", "DIG"])
        preferred_zone = cls.ROLE_PREFERRED_ZONES.get(role, Zone.ANY)

        for task_type in task_order:
            candidates: List[Tuple[int, int, Pos]] = []

            for pos in unlocked_positions:
                if pos in assigned_targets:
                    continue
                
                # 🔥 PERBAIKAN KRITIS: Jangan pernah memilih posisi sendiri sebagai "target baru"
                # Jika worker sudah di sana tapi check_immediate_tile_action gagal, 
                # berarti kondisi tile tidak memungkinkan. Cari tempat lain!
                if pos == unit_pos:
                    continue

                if pos in occupied_tiles and pos != unit_pos:
                    continue

                if cls.is_tile_valid_for_task(pos, state, task_type):
                    d = PathFinder.manhattan_distance(unit_pos, pos)

                    zone_penalty = 0
                    if task_type not in ("HARVEST", "FEED", "COLLECT_FERTILIZER"):
                        tile_zone = cls.get_tile_zone(pos, state)
                        if preferred_zone != Zone.ANY and tile_zone != preferred_zone:
                            zone_penalty = 1

                    if task_type == "FERTILIZE":
                        t = state.tiles[pos[1]][pos[0]]
                        crop = str(t.get("crop", "")) if isinstance(t, dict) else ""
                        prio = 0 if crop == "MELON" else (1 if crop == "STRAWBERRY" else 2)
                        candidates.append((zone_penalty + prio, d, pos))
                    elif task_type == "WATER":
                        t = state.tiles[pos[1]][pos[0]]
                        critical = (
                            int(t.get("consecutive_unwatered", 0))
                            if isinstance(t, dict)
                            else 0
                        )
                        candidates.append((zone_penalty - min(1, critical), d, pos))
                    else:
                        candidates.append((zone_penalty, d, pos))

            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1]))
                _, _, best_target = candidates[0]
                return best_target, task_type

        return None, None