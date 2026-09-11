"""Entrypoint module for Kaggle Environments competition integration."""

from __future__ import annotations

from typing import Any, Dict
from planner import AgentPlanner

_planner_instance = AgentPlanner()


def agent(obs: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
    """Callable entrypoint for Kaggle Environment engine.

    Args:
        obs: Observation state dictionary passed by environment.
        configuration: Optional runtime configurations.

    Returns:
        Structured action payload dictionary.
    """
    return _planner_instance.plan_turn(obs)