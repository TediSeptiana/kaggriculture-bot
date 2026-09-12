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
    regrow_interval: int = 1


CROP_SPECS: Dict[str, CropSpec] = {
    "WHEAT": CropSpec(
        seed_cost=10.0,
        first_yield_day=2,
        max_yield_day=4,
        product_name="WHEAT",
        base_yield=1,
        max_yield=3,
    ),
    "CARROT": CropSpec(20.0, 2, 3, "CARROT", 1, 4),
    "TOMATO": CropSpec(50.0, 8, 11, "TOMATO", 1, 4, "regrow", 3),
    "STRAWBERRY": CropSpec(100.0, 10, 16, "STRAWBERRY", 1, 4, "regrow", 3),
    "MELON": CropSpec(80.0, 10, 10, "MELON", 1, 6),
}

MARKET_PARAMS: Dict[str, Dict[str, float | str]] = {
    "WHEAT": {"base": 25.0, "I0": 100.0, "T": 100.0, "below_func": "linear", "below_target": 2.0, "above_func": "linear", "above_target": 0.5},
    "CARROT": {"base": 35.0, "I0": 80.0, "T": 80.0, "below_func": "linear", "below_target": 2.0, "above_func": "linear", "above_target": 0.5},
    "TOMATO": {"base": 60.0, "I0": 60.0, "T": 60.0, "below_func": "linear", "below_target": 2.0, "above_func": "linear", "above_target": 0.5},
    "STRAWBERRY": {"base": 120.0, "I0": 40.0, "T": 40.0, "below_func": "linear", "below_target": 2.0, "above_func": "linear", "above_target": 0.5},
    "MELON": {"base": 250.0, "I0": 30.0, "T": 30.0, "below_func": "linear", "below_target": 2.0, "above_func": "linear", "above_target": 0.5},
}
SHOP_DEMAND: Dict[str, Dict[str, float]] = {}
TOWN_CENTER_PRODUCTS = set(CROP_SPECS)


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
    market_inventory: Dict[str, int]
    unlocked_shops: List[str]

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
        inventories = []
        for item in raw_inventories:
            if isinstance(item, dict):
                inventories.append(
                    [{str(name): int(count)} for name, count in item.items() if int(count) > 0]
                )
            elif isinstance(item, list):
                inventories.append(item)
            else:
                inventories.append([])

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
            seeds={str(name): int(count) for name, count in (private.get("seeds") or {}).items()},
            inventories=inventories,
            market_prices={
                str(name): float(price)
                for name, price in (market.get("prices") or {}).items()
            },
            market_inventory={
                str(name): int(count)
                for name, count in (market.get("inventory") or {}).items()
            },
            unlocked_shops=list(obs.get("town", {}).keys()) if isinstance(obs.get("town"), dict) else [],
        )

    def remaining_days(self) -> int:
        """Return whole season days remaining, including the current day."""
        return max(0, 30 - self.day)

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

    def planted_counts(self) -> Dict[str, int]:
        """Count planted crops so empty tiles can maintain diversification targets."""
        counts = {crop: 0 for crop in CROP_SPECS}
        for row in self.tiles:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    crop = str(tile.get("crop", ""))
                    if crop in counts:
                        counts[crop] += 1
        return counts

    def planting_targets(self) -> Dict[str, int]:
        """Allocate field capacity across crops instead of monocropping by price."""
        ratios = {
            "WHEAT": 0.20,
            "CARROT": 0.25,
            "TOMATO": 0.15,
            "STRAWBERRY": 0.15,
            "MELON": 0.25,
        }
        tile_count = len(self.get_unlocked_tiles())
        return {crop: max(1, round(tile_count * ratio)) for crop, ratio in ratios.items()}

    @staticmethod
    def is_shed_adjacent(pos: Pos) -> bool:
        """Whether a unit is on one of the simulator's shed access tiles."""
        return pos in {(4, 4), (5, 4), (4, 5), (5, 5)}
