"""Target selection, persistence, and fallback task assignment solver."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from state import CROP_SPECS, FarmState, Pos, Tile
from worker.pathfinding import PathFinder
from worker.roles import WorkerRole


class TaskAssigner:
    """Finds nearest task targets with sticky commitments, primary role priority, and dynamic fallback."""

    # FIX 1: HARVEST selalu nomor 1 untuk SEMUA role
    # Water nomor 2 (kritis, kalau tidak tanaman mati)
    # Plant nomor 3 (bisa nunggu)
    ROLE_HIERARCHY: Dict[WorkerRole, List[str]] = {
        WorkerRole.DIGGER: ["HARVEST", "WATER", "DIG", "PLANT"],
        WorkerRole.PLANTER: ["HARVEST", "WATER", "PLANT", "DIG"],
        WorkerRole.WATERER: ["HARVEST", "WATER", "DIG", "PLANT"],
        WorkerRole.HARVESTER: ["HARVEST", "WATER", "PLANT", "DIG"],
        WorkerRole.ANIMAL: [
            "FEED",
            "PLACE",
            "BUILD",
            "COLLECT_FERTILIZER",
            "HARVEST",
            "WATER",
            "PLANT",
            "DIG",
        ],
        WorkerRole.VERSATILE: ["HARVEST", "WATER", "PLANT", "DIG"],
        # FIX BARU: role FERTILIZE — tugas utama apply fertilizer
        WorkerRole.FERTILIZE: ["FERTILIZE", "HARVEST", "WATER", "PLANT", "DIG"],
    }

    # FIX 2: prioritas crop untuk PLANT (melon > carrot > wheat)
    CROP_PRIORITY: List[str] = ["MELON", "CARROT", "WHEAT", "TOMATO", "STRAWBERRY"]

    @classmethod
    def choose_plant_crop(cls, state: FarmState) -> Optional[str]:
        """Pilih crop berdasarkan prioritas dan seed yang tersedia."""
        for crop in cls.CROP_PRIORITY:
            if state.seeds.get(crop, 0) > 0:
                return crop
        # fallback: seed apapun yang ada
        for crop, count in state.seeds.items():
            if count > 0:
                return crop
        return None

    @classmethod
    def choose_place_animal(cls, state: FarmState) -> Optional[str]:
        """Pilih animal berdasarkan apa yang ada di shed."""
        # Prioritas: cow > sheep > goose (harga produk)
        for animal in ["COW", "SHEEP", "GOOSE"]:
            if state.shed.get(animal, 0) > 0:
                return animal
        return None

    @classmethod
    def is_tile_valid_for_task(
        cls, pos: Pos, state: FarmState, task_type: str
    ) -> bool:
        tx, ty = pos
        tiles = state.tiles
        if ty >= len(tiles) or tx >= len(tiles[ty]):
            return False

        t = tiles[ty][tx]

        if task_type == "DIG":
            return isinstance(t, dict) and t.get("kind") == "WEED"

        if task_type == "PLANT":
            # FIX 3: cek ada seed DAN pilih crop tersedia
            if t is not None:
                return False
            return cls.choose_plant_crop(state) is not None

        if task_type == "PLACE":
            if not (isinstance(t, dict) and t.get("kind") in {"COOP", "PASTURE"}):
                return False
            if t.get("animal"):
                return False
            # FIX 4: hanya PLACE kalau ada animal di shed
            kind = t.get("kind")
            if kind == "COOP":
                return state.shed.get("GOOSE", 0) > 0
            if kind == "PASTURE":
                return state.shed.get("COW", 0) > 0 or state.shed.get("SHEEP", 0) > 0
            return False

        if task_type == "BUILD":
            # FIX 5: hanya build kalau ada animal di shed
            has_coop = any(
                isinstance(tile, dict) and tile.get("kind") == "COOP"
                for row in state.tiles for tile in row
            )
            has_pasture = any(
                isinstance(tile, dict) and tile.get("kind") == "PASTURE"
                for row in state.tiles for tile in row
            )
            needs_coop = (
                not has_coop
                and state.shed.get("GOOSE", 0) > 0
                and t is None
            )
            needs_pasture = (
                not has_pasture
                and (state.shed.get("COW", 0) > 0 or state.shed.get("SHEEP", 0) > 0)
                and t is None
            )
            return needs_coop or needs_pasture

        if task_type == "WATER":
            return (
                isinstance(t, dict)
                and t.get("kind") == "PLANT"
                and not t.get("watered_today", False)
            )

        if task_type == "FEED":
            return (
                isinstance(t, dict)
                and t.get("kind") in {"COOP", "PASTURE"}
                and t.get("animal")
                and not t.get("fed_today", False)
            )

        if task_type == "HARVEST":
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                yield_units = int(t.get("yield_units", 0))
                crop = str(t.get("crop", ""))
                planted_day = int(t.get("planted_day", 0))
                crop_age = state.day - planted_day
                spec = CROP_SPECS.get(crop)
                first_yield = spec.first_yield_day if spec else 2
                return crop_age >= first_yield and yield_units > 0
            # animal harvest (egg/milk/wool)
            if isinstance(t, dict) and t.get("kind") in {"COOP", "PASTURE"}:
                return int(t.get("yield_units", 0)) > 0

        # FIX BARU: validasi tile untuk FERTILIZE
        # Hanya crop premium di bonus window yang belum di-fertilize
        if task_type == "FERTILIZE":
            if not isinstance(t, dict) or t.get("kind") != "PLANT":
                return False
            crop = str(t.get("crop", ""))
            spec = CROP_SPECS.get(crop)
            if spec is None:
                return False
            # Hanya crop premium yang layak fertilizer
            if crop not in ("MELON", "STRAWBERRY", "TOMATO"):
                return False
            planted_day = int(t.get("planted_day", 0))
            age = state.day - planted_day
            fert_until = int(t.get("fertilized_until_day", -1))
            # Sudah di-fertilize? Skip
            if fert_until >= state.day:
                return False
            # Di bonus window?
            bonus_start = (spec.max_yield_day + 1) // 2
            if not (bonus_start <= age <= spec.max_yield_day):
                return False
            return True

        return False

    @classmethod
    def is_target_still_valid(
        cls, pos: Pos, state: FarmState, role: WorkerRole
    ) -> bool:
        task_order = cls.ROLE_HIERARCHY.get(role, ["HARVEST", "WATER", "PLANT", "DIG"])
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
        # 1. Evaluate committed target — TAPI dengan override harvest
        if current_committed_target is not None:
            if (
                current_committed_target not in assigned_targets
                and cls.is_target_still_valid(current_committed_target, state, role)
            ):
                # FIX 6: cek apakah ada HARVEST mendesak di committed
                committed_is_harvest = cls.is_tile_valid_for_task(
                    current_committed_target, state, "HARVEST"
                )
                if committed_is_harvest:
                    return current_committed_target, "HARVEST"

                # Cari harvest terdekat
                unlocked_positions = state.get_unlocked_tiles()
                best_harvest = None
                best_harvest_dist = 9999
                for pos in unlocked_positions:
                    if pos in assigned_targets:
                        continue
                    if cls.is_tile_valid_for_task(pos, state, "HARVEST"):
                        d = PathFinder.manhattan_distance(unit_pos, pos)
                        if d < best_harvest_dist:
                            best_harvest_dist = d
                            best_harvest = pos

                committed_dist = PathFinder.manhattan_distance(
                    unit_pos, current_committed_target
                )
                if best_harvest is not None and best_harvest_dist <= committed_dist:
                    return best_harvest, "HARVEST"

                # Kalau tidak ada harvest mendesak, tetap ke committed
                task_order = cls.ROLE_HIERARCHY.get(
                    role, ["HARVEST", "WATER", "PLANT", "DIG"]
                )
                for task_type in task_order:
                    if cls.is_tile_valid_for_task(current_committed_target, state, task_type):
                        return current_committed_target, task_type

        # 2. Cari target baru dengan scoring
        unlocked_positions = state.get_unlocked_tiles()
        task_order = cls.ROLE_HIERARCHY.get(role, ["HARVEST", "WATER", "PLANT", "DIG"])

        for task_type in task_order:
            candidates: List[Tuple[int, int, Pos]] = []

            for pos in unlocked_positions:
                if pos in assigned_targets:
                    continue

                if cls.is_tile_valid_for_task(pos, state, task_type):
                    d = PathFinder.manhattan_distance(unit_pos, pos)

                    # FIX BARU: prioritas MELON untuk FERTILIZE
                    if task_type == "FERTILIZE":
                        t = state.tiles[pos[1]][pos[0]]
                        crop = str(t.get("crop", "")) if isinstance(t, dict) else ""
                        if crop == "MELON":
                            prio = 0  # paling prioritas
                        elif crop == "STRAWBERRY":
                            prio = 1
                        else:
                            prio = 2
                        candidates.append((prio, d, pos))
                    else:
                        candidates.append((0, d, pos))

            if candidates:
                # Sort by (prioritas, jarak) — prioritas dulu, baru jarak
                candidates.sort(key=lambda x: (x[0], x[1]))
                _, _, best_target = candidates[0]
                return best_target, task_type

        return None, None