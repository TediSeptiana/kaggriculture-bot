"""Domain state and static crop specifications for the farm agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

Pos = Tuple[int, int]
Tile = Dict[str, Any]


@dataclass(frozen=True, slots=True)
class CropSpec:
    """Static lifecycle data for a supported crop."""

    seed_cost: float
    first_yield_day: int
    max_yield_day: int
    product_name: str
    base_yield: int = 1
    max_yield: int = 1
    crop_type: str = "one_time"


# The first strategy deliberately supports wheat only.
CROP_SPECS: Dict[str, CropSpec] = {
    "WHEAT": CropSpec(
        seed_cost=10.0,
        first_yield_day=2,
        max_yield_day=4,
        product_name="WHEAT",
        base_yield=1,
        max_yield=3,
    ),
}


@dataclass(slots=True)
class FarmState:
    """Normalized farm observation used by planning modules."""

    player: int
    day: int
    hour: int
    money: float
    unlocked_quadrants: List[str]
    farmer_pos: Pos
    hands_pos: List[Pos]
    hires_today: int
    tiles: List[List[Any]]
    shed: Dict[str, int]
    seeds: Dict[str, int]
    inventories: List[List[Dict[str, Any]]]
    market_prices: Dict[str, float]

    @classmethod
    def from_obs(cls, obs: Dict[str, Any]) -> "FarmState":
        """Build a normalized state from a Kaggle observation payload."""
        player = int(obs.get("player", 0))
        farms = obs.get("farms") or []
        farm = farms[player] if 0 <= player < len(farms) else {}
        private = obs.get("private") or {}
        market = obs.get("market") or {}

        def position(value: Any, default: Pos = (0, 0)) -> Pos:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return int(value[0]), int(value[1])
            return default

        raw_inventories = private.get("inventories") or []
        inventories = [item if isinstance(item, list) else [] for item in raw_inventories]

        return cls(
            player=player,
            day=int(obs.get("day", 0)),
            hour=int(obs.get("hour", 0)),
            money=float(farm.get("money", 0.0)),
            unlocked_quadrants=list(farm.get("unlocked_quadrants") or []),
            farmer_pos=position(farm.get("farmer")),
            hands_pos=[position(hand) for hand in (farm.get("hands") or [])],
            hires_today=int(farm.get("hires_today", 0)),
            tiles=list(farm.get("tiles") or []),
            shed=dict(private.get("shed") or {}),
            seeds={
                "WHEAT": int((private.get("seeds") or {}).get("WHEAT", 0))
            },
            inventories=inventories,
            market_prices={
                str(name): float(price)
                for name, price in (market.get("prices") or {}).items()
            },
        )

    @staticmethod
    def get_quadrant(x: int, y: int) -> str:
        """Return the quadrant name for a 10x10 farm grid."""
        return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")

    def get_unlocked_tiles(self) -> List[Pos]:
        """Return coordinates inside unlocked quadrants."""
        positions: List[Pos] = []
        for y, row in enumerate(self.tiles):
            for x in range(len(row)):
                if self.get_quadrant(x, y) in self.unlocked_quadrants:
                    positions.append((x, y))
        return positions

    @staticmethod
    def is_shed_adjacent(pos: Pos) -> bool:
        """Whether a unit is on or next to the shed at (4, 4)."""
        return abs(pos[0] - 4) + abs(pos[1] - 4) <= 1
