"""Tests for market order generation."""

from __future__ import annotations

from market import MarketPlanner
from state import FarmState


def test_market_buy_land(dummy_obs: dict) -> None:
    planner = MarketPlanner()
    state = FarmState.from_obs(dummy_obs)
    orders = planner.plan_orders(state)
    assert ["BUY_LAND"] in orders