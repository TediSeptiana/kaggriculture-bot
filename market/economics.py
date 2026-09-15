"""Economic calculations used by crop procurement decisions."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from state import CROP_SPECS, CropSpec, FarmState, MARKET_PARAMS


def demand_coefficients(product: str) -> Tuple[float, float, str]:
    """Return ``a``, ``b`` and curve type for the glut-side demand curve."""
    params = MARKET_PARAMS.get(product, {})
    base = float(params.get("base", 25.0))
    target = float(params.get("above_target", 0.5))
    scale = max(1.0, float(params.get("T", 1.0)))
    curve = str(params.get("above_func", "linear"))
    if curve == "sq":
        return base, target * base / (scale * scale), "sq"
    return base, target * base / scale, "linear"


def price_from_quantity(product: str, quantity: float) -> float:
    """Estimate price from total quantity above the target inventory."""
    a, b, curve = demand_coefficients(product)
    pressure = max(0.0, float(quantity))
    if curve == "sq":
        return max(1.0, a - b * pressure * pressure)
    return max(1.0, a - b * pressure)


def marginal_cost(crop: str) -> float:
    """Seed cost per expected unit of the crop's first production cycle."""
    spec = CROP_SPECS.get(crop)
    if spec is None:
        return float("inf")
    return float(spec.seed_cost) / max(1.0, float(spec.max_yield))


def nash_quantity(
    product: str,
    town_demand: float = 0.0,
    mc: float | None = None,
) -> float:
    """Symmetric two-producer Cournot quantity for the glut-side curve."""
    a, b, curve = demand_coefficients(product)
    marginal = marginal_cost(product) if mc is None else float(mc)
    surplus = max(0.0, a - marginal + b * max(0.0, town_demand))
    if curve == "sq":
        return math.sqrt(surplus / max(1e-9, 8.0 * b))
    return surplus / max(1e-9, 3.0 * b)


def expected_production(spec: CropSpec, remaining_days: int) -> float:
    """Estimate units produced by one seed through the end of season."""
    if remaining_days < spec.first_yield_day:
        return 0.0
    if spec.crop_type == "one_time":
        return float(spec.max_yield)
    events = 1 + max(
        0,
        (remaining_days - spec.first_yield_day) // max(1, spec.regrow_interval),
    )
    return float(min(spec.max_yield, events))


def expected_profit(crop: str, state: FarmState, seed_count: int = 1) -> float:
    """Estimate revenue minus seed cost after adding production pressure."""
    spec = CROP_SPECS.get(crop)
    if spec is None:
        return -float("inf")
    production = expected_production(spec, state.remaining_days()) * seed_count
    if production <= 0:
        return -float("inf")

    product = spec.product_name
    current_inventory = float((state.market_inventory or {}).get(product, 0))
    target = float(MARKET_PARAMS.get(product, {}).get("I0", 0.0))
    pressure = max(0.0, current_inventory + production - target)
    price = price_from_quantity(product, pressure)
    revenue = production * price
    return revenue - float(spec.seed_cost) * seed_count


def rank_crops(state: FarmState, available_seeds: int = 1) -> List[Dict[str, float | str]]:
    """Rank crops by expected profit per seed, including market pressure."""
    ranked: List[Dict[str, float | str]] = []
    for crop, spec in CROP_SPECS.items():
        profit = expected_profit(crop, state, available_seeds)
        ranked.append(
            {
                "crop": crop,
                "profit": profit,
                "profit_per_seed": profit / max(1, available_seeds),
                "mc_per_unit": marginal_cost(crop),
                "nash_quantity": nash_quantity(spec.product_name),
            }
        )
    return sorted(ranked, key=lambda item: float(item["profit"]), reverse=True)