"""Tests for worker action planning and navigation."""

from __future__ import annotations

from state import FarmState
from worker import WorkerPlanner


def test_worker_drop_at_shed(dummy_obs: dict) -> None:
    planner = WorkerPlanner()
    state = FarmState.from_obs(dummy_obs)
    action = planner.decide_action((4, 4), state, [{"kind": "WHEAT"}])
    assert action == ["DROP"]


def test_worker_clear_weed(dummy_obs: dict) -> None:
    planner = WorkerPlanner()
    dummy_obs["farms"][0]["tiles"][4][4] = {"kind": "WEED"}
    state = FarmState.from_obs(dummy_obs)
    action = planner.decide_action((4, 4), state, [])
    assert action == ["DIG"]