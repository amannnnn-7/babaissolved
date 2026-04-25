"""Type definitions for the Baba engine.

We use simple string Enums (not dataclasses-with-id) because:
  - they serialize trivially to JSON for the OpenEnv contract,
  - they round-trip into the LLM prompt as readable tokens,
  - the entire grid state hashes cheaply for BFS visited-sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    WAIT = "wait"

    @property
    def delta(self) -> tuple[int, int]:
        """Return (dy, dx). y grows downward, x grows rightward."""
        return {
            Direction.UP: (-1, 0),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
            Direction.RIGHT: (0, 1),
            Direction.WAIT: (0, 0),
        }[self]


class EntityKind(str, Enum):
    """Physical entities that live on the grid (the *referents*)."""

    BABA = "baba"
    ROCK = "rock"
    WALL = "wall"
    FLAG = "flag"
    SKULL = "skull"
    LAVA = "lava"
    KEKE = "keke"
    DOOR = "door"
    KEY = "key"


class WordKind(str, Enum):
    """Text blocks that, when arranged NOUN-IS-PROPERTY, form rules."""

    # Nouns (use-mention: the WORD baba is distinct from the ENTITY baba)
    W_BABA = "BABA"
    W_ROCK = "ROCK"
    W_WALL = "WALL"
    W_FLAG = "FLAG"
    W_SKULL = "SKULL"
    W_LAVA = "LAVA"
    W_KEKE = "KEKE"
    W_DOOR = "DOOR"
    W_KEY = "KEY"

    # Verb
    W_IS = "IS"

    # Properties
    W_YOU = "YOU"
    W_WIN = "WIN"
    W_STOP = "STOP"
    W_PUSH = "PUSH"
    W_KILL = "KILL"
    W_DEFEAT = "DEFEAT"
    W_SINK = "SINK"
    W_MELT = "MELT"
    W_HOT = "HOT"


# Convenience sets ----------------------------------------------------------

NOUN_WORDS: dict[WordKind, EntityKind] = {
    WordKind.W_BABA: EntityKind.BABA,
    WordKind.W_ROCK: EntityKind.ROCK,
    WordKind.W_WALL: EntityKind.WALL,
    WordKind.W_FLAG: EntityKind.FLAG,
    WordKind.W_SKULL: EntityKind.SKULL,
    WordKind.W_LAVA: EntityKind.LAVA,
    WordKind.W_KEKE: EntityKind.KEKE,
    WordKind.W_DOOR: EntityKind.DOOR,
    WordKind.W_KEY: EntityKind.KEY,
}

VERB_WORDS: set[WordKind] = {WordKind.W_IS}


class Verb(str, Enum):
    IS = "IS"


class Property(str, Enum):
    YOU = "YOU"
    WIN = "WIN"
    STOP = "STOP"
    PUSH = "PUSH"
    KILL = "KILL"
    DEFEAT = "DEFEAT"
    SINK = "SINK"
    MELT = "MELT"
    HOT = "HOT"


PROPERTY_WORDS: dict[WordKind, Property] = {
    WordKind.W_YOU: Property.YOU,
    WordKind.W_WIN: Property.WIN,
    WordKind.W_STOP: Property.STOP,
    WordKind.W_PUSH: Property.PUSH,
    WordKind.W_KILL: Property.KILL,
    WordKind.W_DEFEAT: Property.DEFEAT,
    WordKind.W_SINK: Property.SINK,
    WordKind.W_MELT: Property.MELT,
    WordKind.W_HOT: Property.HOT,
}


# A Tile holds a stack of objects. We encode each object as a single token
# string so the on-the-wire representation is tiny and the LLM sees clean text.
@dataclass(frozen=True, slots=True)
class Tile:
    entities: tuple[EntityKind, ...] = ()
    words: tuple[WordKind, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.words

    def render(self) -> str:
        """ASCII glyph for the LLM-readable grid."""
        if self.words:
            # Show the word in CAPS so use-mention is visually obvious.
            w = self.words[0]
            # For long words use first 3 chars to keep grid aligned.
            return w.value[:3]
        if self.entities:
            e = self.entities[0]
            return {
                EntityKind.BABA: " b ",
                EntityKind.ROCK: " r ",
                EntityKind.WALL: "###",
                EntityKind.FLAG: " F ",
                EntityKind.SKULL: " S ",
                EntityKind.LAVA: "~~~",
                EntityKind.KEKE: " k ",
                EntityKind.DOOR: " D ",
                EntityKind.KEY: " K ",
            }.get(e, " ? ")
        return " . "
