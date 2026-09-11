"""Domain state representations and helper models for Kaggriculture state observation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Pos = Tuple[int, int]
Tile = Optional[Dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CropSpec:
    """Crop economic and lifecycle specification."""

    seed_cost: int
    first_yield_day: int
    max_yield_day: int
    crop_type: str


CROP_SPECS: Dict[str, CropSpec] = {
    "WHEAT": CropSpec(seed_cost=10, first_yield_day=2, max_yield_day=4, crop_type="one_time"),
    "CARROT": CropSpec(seed_cost=20, first_yield_day=2, max_yield_day=3, crop_type="one_time"),
    "TOMATO": CropSpec(seed_cost=50, first_yield_day=8, max_yield_day=11, crop_type="ongoing"),
    "STRAWBERRY": CropSpec(seed_cost=100, first_yield_day=10, max_yield_day=16, crop_type="ongoing"),
    "MELON": CropSpec(seed_cost=80, first_yield_day=10, max_yield_day=10, crop_type="one_time"),
}

SHED_LOCATIONS: Tuple[Pos, ...] = ((4, 4), (5, 4), (4, 5), (5, 5))


@dataclass(slots=True)
class FarmState:
    """Domain model wrapper around the observation state payload."""

    player: int
    day: int
    hour: int
    money: float
    unlocked_quadrants: List[str]
    farmer_pos: Pos
    hands_pos: List[Pos]
    hires_today: int
    tiles: List[List[Tile]]
    shed: Dict[str, int]
    seeds: Dict[str, int]
    inventories: List[List[Dict[str, Any]]]
    market_prices: Dict[str, float]
    market_inventory: Dict[str, int]

    @classmethod
    def from_obs(cls, obs: Dict[str, Any]) -> FarmState:
        """Parses a raw observation dictionary into a typed FarmState instance."""
        player = int(obs["player"])
        farm = obs["farms"][player]
        private = obs.get("private", {})
        market = obs.get("market", {})

        raw_farmer = farm.get("farmer", [0, 0])
        farmer_pos: Pos = (int(raw_farmer[0]), int(raw_farmer[1]))

        hands_pos: List[Pos] = [
            (int(h[0]), int(h[1])) for h in farm.get("hands", [])
        ]

        return cls(
            player=player,
            day=int(obs.get("day", 0)),
            hour=int(obs.get("hour", 0)),
            money=float(farm.get("money", 0.0)),
            unlocked_quadrants=list(farm.get("unlocked_quadrants", ["NW"])),
            farmer_pos=farmer_pos,
            hands_pos=hands_pos,
            hires_today=int(farm.get("hires_today", 0)),
            tiles=farm.get("tiles", []),
            shed=dict(private.get("shed", {})),
            seeds=dict(private.get("seeds", {})),
            inventories=list(private.get("inventories", [[]])),
            market_prices=dict(market.get("prices", {})),
            market_inventory=dict(market.get("inventory", {})),
        )

    def is_shed_adjacent(self, pos: Pos) -> bool:
        """Checks if coordinate matches one of the four center shed tiles."""
        return pos in SHED_LOCATIONS

    @staticmethod
    def get_quadrant(x: int, y: int) -> str:
        """Translates 2D spatial coordinate into quadrant identifier."""
        if x < 5 and y < 5:
            return "NW"
        if x >= 5 and y < 5:
            return "NE"
        if x < 5 and y >= 5:
            return "SW"
        return "SE"

    def get_unlocked_tiles(self) -> List[Pos]:
        """Returns coordinate list for all unlocked tiles."""
        unlocked_set = set(self.unlocked_quadrants)
        return [
            (x, y)
            for y in range(10)
            for x in range(10)
            if self.get_quadrant(x, y) in unlocked_set
        ]