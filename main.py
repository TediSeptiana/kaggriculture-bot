"""Main agent module for Kaggriculture competition."""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

# Type Aliases for high type-safety compliance
Tile = Optional[Union[str, Dict[str, Any]]]
ActionDict = Dict[str, Any]
MarketOrder = List[Union[str, int]]
Pos = Tuple[int, int]


class KaggricultureAgent:
    """Production-grade agent implementing optimal pathfinding and farming economics."""

    CROP_SPECS = {
        "WHEAT": {"seed_cost": 10, "first_yield_day": 2, "max_yield_day": 4, "type": "one_time"},
        "CARROT": {"seed_cost": 20, "first_yield_day": 2, "max_yield_day": 3, "type": "one_time"},
        "TOMATO": {"seed_cost": 50, "first_yield_day": 8, "max_yield_day": 11, "type": "ongoing"},
        "STRAWBERRY": {"seed_cost": 100, "first_yield_day": 10, "max_yield_day": 16, "type": "ongoing"},
        "MELON": {"seed_cost": 80, "first_yield_day": 10, "max_yield_day": 10, "type": "one_time"},
    }

    SHED_LOCATIONS: List[Pos] = [(4, 4), (5, 4), (4, 5), (5, 5)]

    def __init__(self) -> None:
        self.assigned_targets: Dict[int, Pos] = {}

    def is_shed_adjacent(self, pos: Pos) -> bool:
        """Determines if a unit is standing on one of the shed tiles."""
        return pos in self.SHED_LOCATIONS

    def get_quadrant(self, x: int, y: int) -> str:
        """Determines quadrant identifier based on grid coordinates."""
        if x < 5 and y < 5:
            return "NW"
        if x >= 5 and y < 5:
            return "NE"
        if x < 5 and y >= 5:
            return "SW"
        return "SE"

    def get_unlocked_tiles(self, farm: Dict[str, Any]) -> List[Pos]:
        """Gets all accessible coordinate positions on current unlocked land."""
        unlocked_quads = set(farm.get("unlocked_quadrants", ["NW"]))
        return [
            (x, y)
            for y in range(10)
            for x in range(10)
            if self.get_quadrant(x, y) in unlocked_quads
        ]

    def manhattan_distance(self, pos1: Pos, pos2: Pos) -> int:
        """Calculates Manhattan distance between two tile coordinates."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_step_towards(self, current: Pos, target: Pos) -> str:
        """Generates movement directional command to move towards a target coordinate."""
        cx, cy = current
        tx, ty = target
        if cx < tx:
            return "EAST"
        if cx > tx:
            return "WEST"
        if cy < ty:
            return "SOUTH"
        if cy > ty:
            return "NORTH"
        return "PASS"

    def plan_market_actions(
        self,
        farm: Dict[str, Any],
        private: Dict[str, Any],
        market: Dict[str, Any],
        day: int,
        hour: int,
    ) -> List[MarketOrder]:
        """Formulates optimal market queue (orders, hiring, land purchases, and seed trading)."""
        orders: List[MarketOrder] = []
        money: float = farm["money"]
        unlocked_quads = farm.get("unlocked_quadrants", ["NW"])
        seeds = private.get("seeds", {})
        shed = private.get("shed", {})

        # Automatic Land Expansion
        if "NE" not in unlocked_quads and money >= 1200:
            orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked_quads and money >= 2500 and "NE" in unlocked_quads:
            orders.append(["BUY_LAND"])
            money -= 2000

        # Direct Market Selling of Products in Shed
        for item, count in shed.items():
            if count > 0 and item != "WHEAT_SEED":
                orders.append(["SELL", item, count])

        # Dynamic Seed Procurement Strategy
        total_wheat_seeds = seeds.get("WHEAT", 0)
        total_carrot_seeds = seeds.get("CARROT", 0)

        if day < 10:
            if total_wheat_seeds < 10 and money >= 100:
                buy_amount = min(15, int(money // 10))
                if buy_amount > 0:
                    orders.append(["BUY_SEED", "WHEAT", buy_amount])
                    money -= buy_amount * 10
        elif day < 22:
            if total_carrot_seeds < 10 and money >= 200:
                buy_amount = min(10, int(money // 20))
                if buy_amount > 0:
                    orders.append(["BUY_SEED", "CARROT", buy_amount])
                    money -= buy_amount * 20
            if seeds.get("MELON", 0) < 5 and money >= 400:
                buy_amount = min(5, int(money // 80))
                if buy_amount > 0:
                    orders.append(["BUY_SEED", "MELON", buy_amount])
                    money -= buy_amount * 80

        # Hire Farm Hands during peak production periods
        hires_today = farm.get("hires_today", 0)
        if day >= 3 and hour == 0 and hires_today == 0 and money >= 500:
            orders.append(["HIRE"])

        return orders[:10]  # Hard constraint enforcement

    def decide_unit_action(
        self,
        unit_pos: Pos,
        farm: Dict[str, Any],
        private: Dict[str, Any],
        unit_inv: List[Dict[str, Any]],
        day: int,
        hour: int,
    ) -> List[Any]:
        """Generates unit priority actions: Drop -> Clear Weed -> Harvest -> Water -> Plant -> Move."""
        ux, uy = unit_pos
        tiles = farm["tiles"]
        current_tile: Tile = tiles[uy][ux]
        unlocked_positions = self.get_unlocked_tiles(farm)

        # 1. Clear inventory at shed if full/carrying products
        if len(unit_inv) > 0 and self.is_shed_adjacent(unit_pos):
            return ["DROP"]

        # 2. Clear Weed
        if isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
            return ["DIG"]

        # 3. Harvest mature crops
        if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
            yield_units = current_tile.get("yield_units", 0)
            crop = current_tile.get("crop")
            planted_day = current_tile.get("planted_day", 0)
            crop_age = day - planted_day
            spec = self.CROP_SPECS.get(crop, {})

            if crop_age >= spec.get("first_yield_day", 2) and yield_units > 0:
                return ["HARVEST"]

            if not current_tile.get("watered_today", False):
                return ["WATER"]

        # 4. Plant new seeds
        if current_tile is None and self.get_quadrant(ux, uy) in farm.get("unlocked_quadrants", ["NW"]):
            seeds = private.get("seeds", {})
            if seeds.get("WHEAT", 0) > 0:
                return ["PLANT", "WHEAT"]
            if seeds.get("CARROT", 0) > 0:
                return ["PLANT", "CARROT"]
            if seeds.get("MELON", 0) > 0:
                return ["PLANT", "MELON"]

        # 5. Pathfinding towards priority task
        best_target: Optional[Pos] = None
        min_dist = 999

        for pos in unlocked_positions:
            tx, ty = pos
            t = tiles[ty][tx]

            needs_action = False
            if t is None and sum(private.get("seeds", {}).values()) > 0:
                needs_action = True
            elif isinstance(t, dict):
                if t.get("kind") == "WEED":
                    needs_action = True
                elif t.get("kind") == "PLANT" and not t.get("watered_today", False):
                    needs_action = True

            if needs_action:
                d = self.manhattan_distance(unit_pos, pos)
                if d < min_dist:
                    min_dist = d
                    best_target = pos

        if best_target:
            move_cmd = self.get_step_towards(unit_pos, best_target)
            if move_cmd != "PASS":
                return [move_cmd]

        return ["PASS"]

    def __call__(self, obs: Dict[str, Any]) -> ActionDict:
        """Main agent entry point called by Kaggle Environment engine."""
        player = obs["player"]
        day = obs["day"]
        hour = obs["hour"]
        farm = obs["farms"][player]
        private = obs["private"]
        market = obs["market"]

        market_actions = self.plan_market_actions(farm, private, market, day, hour)

        farmer_pos: Pos = tuple(farm["farmer"])  # type: ignore
        farmer_inv = private.get("inventories", [[]])[0]
        farmer_action = self.decide_unit_action(farmer_pos, farm, private, farmer_inv, day, hour)

        hands_actions: List[List[Any]] = []
        hands_positions = farm.get("hands", [])
        all_inventories = private.get("inventories", [])

        for idx, hand_pos in enumerate(hands_positions):
            h_pos: Pos = tuple(hand_pos)  # type: ignore
            h_inv = all_inventories[idx + 1] if idx + 1 < len(all_inventories) else []
            h_act = self.decide_unit_action(h_pos, farm, private, h_inv, day, hour)
            hands_actions.append(h_act)

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": market_actions,
        }


# Standard instance export
_agent_instance = KaggricultureAgent()


def agent(obs: Dict[str, Any]) -> ActionDict:
    """Callable framework wrapper for main.py."""
    return _agent_instance(obs)