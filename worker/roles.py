"""Worker specialization roles and task priority definitions."""

from __future__ import annotations

from enum import Enum, auto


class WorkerRole(Enum):
    """Specialized worker role hierarchy."""

    DIGGER = auto()      # Main Farmer: Focuses on DIGWEEDS, but falls back to general field work
    PLANTER = auto()     # Hand 0: Focuses on PLANTING seeds
    WATERER = auto()     # Hand 1: Focuses on WATERING unwatered crops
    HARVESTER = auto()   # Hand 2: Focuses on HARVESTING mature crops
    ANIMAL = auto()      # Builds structures and places livestock
    VERSATILE = auto()   # Generic worker fallback