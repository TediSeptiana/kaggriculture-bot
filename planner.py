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

    def _assess_global_farm_demand(self, state: FarmState) -> List[WorkerRole]:
        """Evaluates macro demand across the farm to create a dynamic role priority list."""
        unlocked = state.get_unlocked_tiles()
        tiles = state.tiles

        unwatered_count = 0
        harvestable_count = 0
        weed_count = 0
        empty_count = 0

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

        # Build demand priority order based on macro urgency
        role_demands: List[WorkerRole] = []

        # Urgent Priority 1: Harvesting mature crops (Direct Cash Liquidation)
        if harvestable_count > 0:
            role_demands.extend([WorkerRole.HARVESTER] * min(3, harvestable_count))

        # Urgent Priority 2: Watering thirsty crops (Prevent Yield Degradation)
        if unwatered_count > 0:
            role_demands.extend([WorkerRole.WATERER] * min(4, unwatered_count))

        # Priority 3: Planting if seeds are available and empty tiles exist
        total_seeds = sum(state.seeds.values())
        if empty_count > 0 and total_seeds > 0:
            role_demands.extend([WorkerRole.PLANTER] * min(2, empty_count))

        # Priority 4: Clearing weeds
        if weed_count > 0:
            role_demands.extend([WorkerRole.DIGGER] * min(2, weed_count))

        # Fallback to VERSATILE if no specific dominant demand
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

        # ------------------------------------------------------------------
        # 1. Market Queue Planning
        # ------------------------------------------------------------------
        market_actions = self.market_planner.plan_orders(state)

        # ------------------------------------------------------------------
        # 2. Dynamic Macro Demand Assessment
        # ------------------------------------------------------------------
        demand_roles = self._assess_global_farm_demand(state)

        # Prevent multiple units from selecting the same tile during this turn.
        assigned_targets: Set[Pos] = set()

        # ------------------------------------------------------------------
        # 3. Animal work state (dipakai farmer & hands)
        # ------------------------------------------------------------------
        has_coop = any(
            isinstance(tile, dict) and tile.get("kind") == "COOP"
            for row in state.tiles for tile in row
        )
        has_pasture = any(
            isinstance(tile, dict) and tile.get("kind") == "PASTURE"
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
        waiting_animal = shed_goose > 0 or shed_cow > 0

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

        # ------------------------------------------------------------------
        # 4. Main Farmer Execution (Unit ID: 0)
        # FIX: Farmer hanya ANIMAL saat TRANSISI (pickup/place), bukan terus-menerus
        # ------------------------------------------------------------------
        farmer_inv = state.inventories[0] if len(state.inventories) > 0 else []
        if self.pending_animal:
            farmer_inv = list(farmer_inv) + [{self.pending_animal: 1}]

        # Farmer jadi ANIMAL HANYA kalau:
        #   - sedang bawa animal (pickup → place)
        #   - ada animal di shed tapi belum ada struktur untuk place
        farmer_needs_animal = (
            self.pending_animal is not None
            or (shed_goose > 0 and not has_coop)
            or (shed_goose > 0 and has_empty_coop)
            or (shed_cow > 0 and not has_pasture)
            or (shed_cow > 0 and has_empty_pasture)
        )

        farmer_role = (
            WorkerRole.ANIMAL
            if farmer_needs_animal
            else (demand_roles[0] if demand_roles else WorkerRole.DIGGER)
        )

        farmer_action = self.worker_planner.decide_action(
            unit_pos=state.farmer_pos,
            state=state,
            inventory=farmer_inv,
            role=farmer_role,
            assigned_targets=assigned_targets,
        )

        # Update pending_animal state
        if farmer_action and farmer_action[0] == "PICKUP":
            self.pending_animal = str(farmer_action[1])
            self.market_planner.pending_animal = self.pending_animal
        elif farmer_action and farmer_action[0] == "PLACE":
            self.pending_animal = None
            self.market_planner.pending_animal = None

        # ------------------------------------------------------------------
        # 5. Hired Hands Execution (Unit IDs: 1..N)
        # FIX: Hand ke-4 (idx 3) jadi ANIMAL specialist
        # ------------------------------------------------------------------
        # Apakah ada pekerjaan animal yang perlu hand?
        animal_work_available = (
            has_hungry_animal
            or has_animal_yield
            or (shed_goose > 0 and has_empty_coop)
            or (shed_cow > 0 and has_empty_pasture)
            or (shed_goose > 0 and not has_coop)
            or (shed_cow > 0 and not has_pasture)
        )

        hands_actions: List[List[Any]] = []
        for idx, hand_pos in enumerate(state.hands_pos):
            hand_unit_id = idx + 1
            hand_inv = (
                state.inventories[hand_unit_id]
                if hand_unit_id < len(state.inventories)
                else []
            )

            # FIX: hand idx 3 = animal specialist
            if idx == 3 and animal_work_available:
                assigned_role = WorkerRole.ANIMAL
            elif idx == 3:
                # Tidak ada animal work, tapi hand ke-4 ada → VERSATILE
                assigned_role = WorkerRole.VERSATILE
            else:
                role_idx = (idx + 1) % len(demand_roles)
                assigned_role = demand_roles[role_idx]

            hand_act = self.worker_planner.decide_action(
                unit_pos=hand_pos,
                state=state,
                inventory=hand_inv,
                role=assigned_role,
                assigned_targets=assigned_targets,
            )
            hands_actions.append(hand_act)

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": market_actions,
        }