"""Integration tests for high-level agent planner."""

from __future__ import annotations

from planner import AgentPlanner


def test_agent_planner_execution(dummy_obs: dict) -> None:
    planner = AgentPlanner()
    actions = planner.plan_turn(dummy_obs)
    assert "farmer" in actions
    assert "hands" in actions
    assert "market" in actions