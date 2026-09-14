"""Worker specialization roles and task priority definitions."""

from __future__ import annotations

from enum import Enum, auto


class WorkerRole(Enum):
    """Specialized worker role hierarchy."""
    DIGGER = auto()
    PLANTER = auto()
    WATERER = auto()
    HARVESTER = auto()
    RANCHER = auto()   # ← baru: feed & collect animal
    ANIMAL = auto()      # Builds structures and places livestock
    VERSATILE = auto()   # Generic worker fallback
    FERTILIZE = auto() 