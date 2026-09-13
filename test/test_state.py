"""Tests for state parsing and domain model validation."""

from __future__ import annotations

import pytest
from state import FarmState


@pytest.fixture
def dummy_obs() -> dict:
    return {
        "player": 0,
        "day": 1,
        "hour": 5,
        "farms": [
            {
                "money": 1500.0,
                "unlocked_quadrants": ["NW"],
                "farmer": [4, 4],
                "hands": [[5, 4]],
                "hires_today": 1,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
            }
        ],
        "private": {
            "shed": {"WHEAT": 5},
            "seeds": {"WHEAT": 2},
            "inventories": [[], []],
        },
        "market": {"prices": {"WHEAT": 25.0}, "inventory": {"WHEAT": 10000}},
    }


def test_farm_state_from_obs(dummy_obs: dict) -> None:
    state = FarmState.from_obs(dummy_obs)
    assert state.player == 0
    assert state.day == 1
    assert state.money == 1500.0
    assert state.farmer_pos == (4, 4)
    assert state.is_shed_adjacent((4, 4)) is True
    assert len(state.get_unlocked_tiles()) == 25


def test_farm_state_uses_active_player_farm(dummy_obs: dict) -> None:
    dummy_obs["player"] = 1
    dummy_obs["farms"].append(
        {
            "money": 2750.0,
            "unlocked_quadrants": ["NE"],
            "farmer": [8, 1],
            "hands": [],
            "hires_today": 0,
            "tiles": [[None for _ in range(10)] for _ in range(10)],
        }
    )

    state = FarmState.from_obs(dummy_obs)

    assert state.player == 1
    assert state.money == 2750.0
    assert state.farmer_pos == (8, 1)
    assert state.unlocked_quadrants == ["NE"]