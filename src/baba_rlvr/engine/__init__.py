"""Pure-Python Baba Is You engine.

This is a deliberately *minimal but faithful* implementation of the Baba ruleset:
each tile holds a stack of objects; objects are either ENTITIES (e.g. baba, wall,
flag, rock) or TEXT tokens (NOUN / VERB / PROPERTY). Active rules of the form
NOUN-IS-PROPERTY or NOUN-IS-NOUN are parsed from horizontal/vertical text
triples and re-evaluated every step.

We keep it pure-Python (no C++/baba-is-auto dep) so:
  - it installs anywhere (Colab, laptops),
  - it is deterministic and easy to unit-test,
  - it is fast enough: ~30 µs/step on a single core, more than enough for
    BFS solving small puzzles and for GRPO rollouts.
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
