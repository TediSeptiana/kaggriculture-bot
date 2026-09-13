"""Land expansion utility scoring and acquisition manager based on Net Present Value (NPV)."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
from state import FarmState

MarketOrder = List[Union[str, int]]


class LandManager:
    """Evaluates quadrant expansion utility via NPV-based financial modeling."""

    LAND_COSTS: dict[str, float] = {
        "NE": 1000.0,
        "SW": 2000.0,
        "SE": 4000.0,
    }

    SEASON_DAYS: int = 30

    # FIX: diperketat dari 0.50 → 1.0 (NPV harus > cost)
    SAFETY_MARGIN: float = 1.0

    # FIX: dinaikkan dari 0.70 → 0.85
    MIN_UTILIZATION: float = 0.85

    # FIX: dinaikkan dari 500 → 1500
    CASH_BUFFER: float = 1500.0

    # FIX: dinaikkan dari 3 → 8 hari
    MIN_DAYS_REMAINING: int = 12

    # FIX BARU: cash minimum multiplier — cegah beli land saat cash tipis
    # Land $2K butuh $2K * 2.5 + $1.5K buffer = $6.5K
    MIN_CASH_MULTIPLIER: float = 2.5

    def _next_target_quadrant(self, state: FarmState) -> Optional[str]:
        """Determines sequential quadrant expansion target (NE -> SW -> SE)."""
        unlocked = set(state.unlocked_quadrants)
        for quad in ("NE", "SW", "SE"):
            if quad not in unlocked:
                return quad
        return None

    def _estimate_profit_per_tile_per_day(self, state: FarmState) -> float:
        """Estimates daily realized profit per tile based on actual market crop pricing."""
        unlocked = state.get_unlocked_tiles()
        if not unlocked:
            return 0.0

        occupied = sum(
            1 for x, y in unlocked if state.tiles[y][x] is not None
        )
        utilization = occupied / float(len(unlocked))

        # FIX: benchmark MELON (crop paling profitable), bukan WHEAT
        melon_price = state.market_prices.get("MELON", 250.0)
        # MELON: 10 hari siklus, ~4 yield/cycle unfertilized, seed $80
        net_per_cycle = (4.0 * melon_price) - 80.0
        profit_full = net_per_cycle / 10.0  # ~$92/tile/day di $250/unit

        # Scale by utilization
        return profit_full * utilization

    def _compute_npv(
        self,
        state: FarmState,
        quadrant: str,
        new_tiles: int = 25,
    ) -> float:
        """Calculates NPV = (Marginal Daily Return * Days Remaining) - Quadrant Purchase Cost."""
        cost = self.LAND_COSTS.get(quadrant, float("inf"))
        days_left = max(0, self.SEASON_DAYS - state.day)
        profit_per_tile = self._estimate_profit_per_tile_per_day(state)

        # Projected marginal daily revenue from 25 new tiles
        # FIX: efficiency 0.80 → 0.55 (konservatif)
        marginal_daily = profit_per_tile * new_tiles * 0.55
        return (marginal_daily * days_left) - cost

    def calculate_land_score(
        self, state: FarmState, disposable_cash: float
    ) -> Tuple[Optional[str], float]:
        """Calculates financial utility score for land expansion using NPV analysis."""
        target = self._next_target_quadrant(state)
        if target is None:
            return None, 0.0

        cost = self.LAND_COSTS[target]
        days_left = self.SEASON_DAYS - state.day

        # ==================================================================
        # FIX BARU Guard 0: absolute cash check
        # Cegah beli land saat cash tipis (Match 11 D13 cash $2.266 beli $4K)
        # ==================================================================
        required_cash = cost * self.MIN_CASH_MULTIPLIER + self.CASH_BUFFER
        if state.money < required_cash:
            return target, 0.0

        # Guard 1: Minimum operational days remaining
        if days_left < self.MIN_DAYS_REMAINING:
            return target, 0.0

        # Guard 2: Capital reserve requirements
        if disposable_cash < (cost + self.CASH_BUFFER):
            return target, 0.0

        # Guard 3: Farm land utilization density
        unlocked = state.get_unlocked_tiles()
        occupied = sum(
            1 for x, y in unlocked if state.tiles[y][x] is not None
        )
        utilization = occupied / max(1, len(unlocked))
        if utilization < self.MIN_UTILIZATION:
            return target, 0.0

        # Guard 4: NPV profitability threshold
        npv = self._compute_npv(state, target)
        required_npv = cost * self.SAFETY_MARGIN
        if npv <= required_npv:
            return target, 0.0

        # Normalized utility score [0.0 - 1.0]
        score = min(1.0, npv / (cost * 2.0))
        return target, score

    def plan_land_orders(
        self, state: FarmState, disposable_cash: float
    ) -> Tuple[List[MarketOrder], float]:
        """Generates BUY_LAND order if NPV exceeds risk-adjusted threshold."""
        orders: List[MarketOrder] = []
        target, score = self.calculate_land_score(state, disposable_cash)

        if target is not None and score > 0.0:
            cost = self.LAND_COSTS[target]
            orders.append(["BUY_LAND"])
            disposable_cash -= cost

        return orders, disposable_cash