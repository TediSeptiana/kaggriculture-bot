"""High-level orchestration agent applying market demand-driven worker dispatching and global target persistence."""

from __future__ import annotations

from typing import Any, Dict, List, Set
from market import MarketPlanner
from state import CROP_SPECS, FarmState
from worker import Pos, WorkerPlanner, WorkerRole


class AgentPlanner:
    """Orchestrates state parsing, global target persistence, and demand-driven worker dispatching."""

    def __init__(self) -> None:
        self.worker_planner = WorkerPlanner()
        self.market_planner = MarketPlanner()
        self.pending_animal: str | None = None
        self.last_day: int | None = None
        self.worker_roles: Dict[int, WorkerRole] = {}

    def _assess_global_farm_demand(self, state: FarmState) -> List[WorkerRole]:
        """Evaluates macro demand across the farm to create a dynamic role priority list."""
        unlocked = state.get_unlocked_tiles()
        tiles = state.tiles

        unwatered_count = 0
        harvestable_count = 0
        weed_count = 0
        empty_count = 0

        has_coop = False
        has_pasture = False
        has_empty_coop = False
        has_empty_pasture = False
        has_hungry_animal = False
        has_animal_yield = False

        for pos in unlocked:
            tx, ty = pos
            if ty >= len(tiles) or tx >= len(tiles[ty]):
                continue
            t = tiles[ty][tx]

            if t is None:
                empty_count += 1
            elif isinstance(t, dict):
                kind = t.get("kind")

                if kind == "WEED":
                    weed_count += 1

                elif kind == "PLANT":
                    if not t.get("watered_today", False):
                        unwatered_count += 1

                    crop = str(t.get("crop", ""))
                    planted_day = int(t.get("planted_day", 0))
                    yield_units = int(t.get("yield_units", 0))
                    crop_age = state.day - planted_day
                    spec = CROP_SPECS.get(crop)
                    first_yield = spec.first_yield_day if spec else 2

                    if crop_age >= first_yield and yield_units > 0:
                        harvestable_count += 1

                elif kind == "COOP":
                    has_coop = True
                    if not t.get("animal"):
                        has_empty_coop = True
                    else:
                        if not t.get("fed_today", False):
                            has_hungry_animal = True
                        if int(t.get("yield_units", 0)) > 0:
                            has_animal_yield = True

                elif kind == "PASTURE":
                    has_pasture = True
                    if not t.get("animal"):
                        has_empty_pasture = True
                    else:
                        if not t.get("fed_today", False):
                            has_hungry_animal = True
                        if int(t.get("yield_units", 0)) > 0:
                            has_animal_yield = True

        role_demands: List[WorkerRole] = []

        # Priority 1: Harvest
        if harvestable_count > 0:
            role_demands.extend([WorkerRole.HARVESTER] * min(5, harvestable_count))

        # Priority 2: Water
        if unwatered_count > 0:
            water_cap = 6 if unwatered_count > 6 else 4
            role_demands.extend([WorkerRole.WATERER] * min(water_cap, unwatered_count))

        # Priority 3: Plant
        total_seeds = sum(state.seeds.values())
        if empty_count > 0 and total_seeds > 0:
            role_demands.extend([WorkerRole.PLANTER] * min(2, empty_count))

        # Priority 4: Clear weeds
        if weed_count > 0:
            digger_cap = 4 if weed_count > 10 else 2
            role_demands.extend([WorkerRole.DIGGER] * min(digger_cap, weed_count))
        elif empty_count == 0 and state.seeds.get("MELON", 0) > 0:
            role_demands.append(WorkerRole.DIGGER)

        # ------------------------------------------------------------------
        # Priority 5: Animal care — BUILD_PASTURE / PLACE / FEED
        # FASE B: fokus cow + sheep (bukan goose)
        # ------------------------------------------------------------------
        shed = getattr(state, "shed", {}) or {}
        # FIX: cek inventory juga untuk trigger BUILD cepat
        inv_cow = 0
        inv_sheep = 0
        inv_goose = 0
        for inv in state.inventories:
            if isinstance(inv, list):
                for entry in inv:
                    if isinstance(entry, dict):
                        inv_cow += int(entry.get("COW", 0))
                        inv_sheep += int(entry.get("SHEEP", 0))
                        inv_goose += int(entry.get("GOOSE", 0))

        has_goose_in_shed = (shed.get("GOOSE", 0) + inv_goose) > 0
        has_cow_in_shed = (shed.get("COW", 0) + inv_cow) > 0
        has_sheep_in_shed = (shed.get("SHEEP", 0) + inv_sheep) > 0

        need_build_coop = has_goose_in_shed and not has_empty_coop
        need_build_pasture = (
            (has_cow_in_shed or has_sheep_in_shed) and not has_empty_pasture
        )
        need_build = need_build_coop or need_build_pasture

        need_place = (has_empty_coop and has_goose_in_shed) or (
            has_empty_pasture and (has_cow_in_shed or has_sheep_in_shed)
        )

        need_feed = has_hungry_animal or has_animal_yield

        if need_build or need_place or need_feed:
            # FASE B: hitung total animal di farm
            animal_count = sum(
                1 for row in tiles for t in row
                if isinstance(t, dict) and t.get("animal")
            )
            if need_feed:
                role_demands.insert(0, WorkerRole.ANIMAL)
                # FASE B: 2+ animal butuh 2 worker
                if animal_count >= 2:
                    role_demands.insert(1, WorkerRole.ANIMAL)
            if need_place:
                role_demands.append(WorkerRole.ANIMAL)
            if need_build:
                role_demands.append(WorkerRole.ANIMAL)

        # Priority 6: Fertilize
        has_fertilizer_in_shed = shed.get("FERTILIZER", 0) > 0
        if has_fertilizer_in_shed:
            need_fertilize = False
            for row in tiles:
                for t in row:
                    if not isinstance(t, dict) or t.get("kind") != "PLANT":
                        continue
                    crop = str(t.get("crop", ""))
                    if crop not in ("MELON", "STRAWBERRY", "TOMATO"):
                        continue
                    spec = CROP_SPECS.get(crop)
                    if spec is None:
                        continue
                    planted_day = int(t.get("planted_day", 0))
                    age = state.day - planted_day
                    fert_until = int(t.get("fertilized_until_day", -1))
                    if fert_until >= state.day:
                        continue
                    bonus_start = (spec.max_yield_day + 1) // 2
                    if bonus_start <= age <= spec.max_yield_day:
                        need_fertilize = True
                        break
                if need_fertilize:
                    break

            if need_fertilize:
                role_demands.insert(0, WorkerRole.FERTILIZE)

        if not role_demands:
            role_demands = [WorkerRole.VERSATILE]

        return role_demands

    def plan_turn(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point processing observation dict and returning action payload."""
        state = FarmState.from_obs(obs)

        if state.day == 0 and self.last_day != 0:
            self.pending_animal = None
            self.market_planner.pending_animal = None
        self.last_day = state.day

        # 1. Market Queue Planning
        market_actions = self.market_planner.plan_orders(state)

        # 2. Dynamic Macro Demand Assessment
        demand_roles = self._assess_global_farm_demand(state)

        assigned_targets: Set[Pos] = set()

        # 3. Animal work state
        has_coop = any(
            isinstance(tile, dict) and tile.get("kind") == "COOP"
            for row in state.tiles for tile in row
        )
        has_empty_coop = any(
            isinstance(tile, dict)
            and tile.get("kind") == "COOP"
            and not tile.get("animal")
            for row in state.tiles for tile in row
        )
        has_empty_pasture = any(
            isinstance(tile, dict)
            and tile.get("kind") == "PASTURE"
            and not tile.get("animal")
            for row in state.tiles for tile in row
        )

        shed_goose = state.shed.get("GOOSE", 0)
        shed_cow = state.shed.get("COW", 0)
        shed_sheep = state.shed.get("SHEEP", 0)
        has_pasture = any(
            isinstance(tile, dict) and tile.get("kind") == "PASTURE"
            for row in state.tiles for tile in row
        )

        # FIX: hitung animal di inventory juga
        inventory_cow = 0
        inventory_sheep = 0
        inventory_goose = 0
        for inv in state.inventories:
            if isinstance(inv, list):
                for entry in inv:
                    if isinstance(entry, dict):
                        inventory_cow += int(entry.get("COW", 0))
                        inventory_sheep += int(entry.get("SHEEP", 0))
                        inventory_goose += int(entry.get("GOOSE", 0))

        total_cow_pending = shed_cow + inventory_cow
        total_sheep_pending = shed_sheep + inventory_sheep
        total_goose_pending = shed_goose + inventory_goose

        has_hungry_animal = any(
            isinstance(tile, dict)
            and tile.get("kind") in ("COOP", "PASTURE")
            and tile.get("animal")
            and not tile.get("fed_today", False)
            for row in state.tiles for tile in row
        )
        has_animal_yield = any(
            isinstance(tile, dict)
            and tile.get("kind") in ("COOP", "PASTURE")
            and tile.get("animal")
            and int(tile.get("yield_units", 0)) > 0
            for row in state.tiles for tile in row
        )

        harvestable_crops = 0
        for row in state.tiles:
            for t in row:
                if isinstance(t, dict) and t.get("kind") == "PLANT":
                    if int(t.get("yield_units", 0)) <= 0:
                        continue
                    crop = str(t.get("crop", ""))
                    planted_day = int(t.get("planted_day", 0))
                    crop_age = state.day - planted_day
                    spec = CROP_SPECS.get(crop)
                    first_yield = spec.first_yield_day if spec else 2
                    if crop_age >= first_yield:
                        harvestable_crops += 1

        urgent_harvest = harvestable_crops >= 3

        # ============================================================
        # 4. Farmer Execution — assign role SEKALI per hari
        # ============================================================
        # ... (kode sebelumnya tetap sama sampai bagian 4. Farmer Execution) ...

        # ============================================================
        # 4. Farmer Execution
        # ============================================================
        farmer_inv = state.inventories[0] if len(state.inventories) > 0 else []
        if self.pending_animal:
            farmer_inv = list(farmer_inv) + [{self.pending_animal: 1}]

        farmer_needs_animal = (
            self.pending_animal is not None
            or (total_goose_pending > 0 and not has_empty_coop)
            or (total_cow_pending > 0 and not has_empty_pasture)
            or (total_sheep_pending > 0 and not has_empty_pasture)
            or has_hungry_animal
        )

        if farmer_needs_animal:
            farmer_role = WorkerRole.ANIMAL
        else:
            farmer_role = demand_roles[0] if demand_roles else WorkerRole.DIGGER

        # 🔥 KUMPULKAN SEMUA POSISI WORKER UNTUK COLLISION AVOIDANCE
        all_worker_positions = {state.farmer_pos} | set(state.hands_pos)

        farmer_action = self.worker_planner.decide_action(
            unit_pos=state.farmer_pos,
            state=state,
            inventory=farmer_inv,
            role=farmer_role,
            assigned_targets=assigned_targets,
            other_worker_positions=all_worker_positions,  # <-- DITAMBAHKAN
        )

        if farmer_action and farmer_action[0] == "PICKUP":
            self.pending_animal = str(farmer_action[1])
            self.market_planner.pending_animal = self.pending_animal
        elif farmer_action and farmer_action[0] == "PLACE":
            self.pending_animal = None
            self.market_planner.pending_animal = None

        # ============================================================
        # 5. Hands Execution
        # ============================================================
        animal_work_available = (
            has_hungry_animal
            or has_animal_yield
            or (total_goose_pending > 0 and has_empty_coop)
            or (total_cow_pending > 0 and has_empty_pasture)
            or (total_sheep_pending > 0 and has_empty_pasture)
            or (total_goose_pending > 0 and not has_coop)
            or (total_cow_pending > 0 and not has_pasture)
            or (total_sheep_pending > 0 and not has_pasture)
        )

        hands_actions: List[List[Any]] = []
        for idx, hand_pos in enumerate(state.hands_pos):
            hand_unit_id = idx + 1

            if idx == 3 and animal_work_available and not urgent_harvest:
                assigned_role = WorkerRole.ANIMAL
            elif idx == 3 and urgent_harvest:
                assigned_role = WorkerRole.HARVESTER
            elif idx == 3:
                assigned_role = WorkerRole.VERSATILE
            else:
                role_idx = idx % len(demand_roles)
                assigned_role = demand_roles[role_idx]
            hand_inv = (
                state.inventories[hand_unit_id]
                if hand_unit_id < len(state.inventories)
                else []
            )

            hand_act = self.worker_planner.decide_action(
                unit_pos=hand_pos,
                state=state,
                inventory=hand_inv,
                role=assigned_role,
                assigned_targets=assigned_targets,
                other_worker_positions=all_worker_positions,  # <-- DITAMBAHKAN
            )
            hands_actions.append(hand_act)

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": market_actions,
        }