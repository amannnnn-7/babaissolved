"""Baba Is You engine — Python adapter over the C++ `pyBaba` library.

Mechanics (push chains, NOUN-IS-NOUN transformations, defeat / sink / hot+melt,
overlap detection) are delegated to `pyBaba` (utilForever/baba-is-auto).
This package wraps that engine in a stable, readable Python API used by the
OpenEnv server, the verifier (RewardTracker), the visualizer, and the BFS
solver.
"""

from .types import (
    Direction,
    EntityKind,
    Property,
    Tile,
    Verb,
    WordKind,
)
from .world import World, parse_level

__all__ = [
    "Direction",
    "EntityKind",
    "Property",
    "Tile",
    "Verb",
    "WordKind",
    "World",
    "parse_level",
]
