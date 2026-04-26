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
    WATER = "water"
    GRASS = "grass"
    TILE = "tile"
    FLOWER = "flower"
    ICE = "ice"
    JELLY = "jelly"
    CRAB = "crab"
    LOVE = "love"
    ALGAE = "algae"
    HEDGE = "hedge"
    BELT = "belt"
    BUG = "bug"
    ROBOT = "robot"
    STAR = "star"


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
    W_WATER = "WATER"
    W_GRASS = "GRASS"
    W_TILE = "TILE"
    W_FLOWER = "FLOWER"
    W_ICE = "ICE"
    W_JELLY = "JELLY"
    W_CRAB = "CRAB"
    W_LOVE = "LOVE"
    W_ALGAE = "ALGAE"
    W_HEDGE = "HEDGE"
    W_BELT = "BELT"
    W_BUG = "BUG"
    W_ROBOT = "ROBOT"
    W_STAR = "STAR"

    # Verbs / operators
    W_IS = "IS"
    W_HAS = "HAS"
    W_MAKE = "MAKE"
    W_AND = "AND"
    W_NOT = "NOT"
    W_ON = "ON"
    W_NEAR = "NEAR"
    W_FACING = "FACING"
    W_LONELY = "LONELY"

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
    W_OPEN = "OPEN"
    W_SHUT = "SHUT"
    W_MOVE = "MOVE"
    W_SHIFT = "SHIFT"
    W_PULL = "PULL"
    W_SWAP = "SWAP"
    W_TELE = "TELE"
    W_FLOAT = "FLOAT"
    W_WEAK = "WEAK"
    W_MORE = "MORE"
    W_SAFE = "SAFE"


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
    WordKind.W_WATER: EntityKind.WATER,
    WordKind.W_GRASS: EntityKind.GRASS,
    WordKind.W_TILE: EntityKind.TILE,
    WordKind.W_FLOWER: EntityKind.FLOWER,
    WordKind.W_ICE: EntityKind.ICE,
    WordKind.W_JELLY: EntityKind.JELLY,
    WordKind.W_CRAB: EntityKind.CRAB,
    WordKind.W_LOVE: EntityKind.LOVE,
    WordKind.W_ALGAE: EntityKind.ALGAE,
    WordKind.W_HEDGE: EntityKind.HEDGE,
    WordKind.W_BELT: EntityKind.BELT,
    WordKind.W_BUG: EntityKind.BUG,
    WordKind.W_ROBOT: EntityKind.ROBOT,
    WordKind.W_STAR: EntityKind.STAR,
}

VERB_WORDS: set[WordKind] = {
    WordKind.W_IS,
    WordKind.W_HAS,
    WordKind.W_MAKE,
    WordKind.W_AND,
    WordKind.W_NOT,
    WordKind.W_ON,
    WordKind.W_NEAR,
    WordKind.W_FACING,
    WordKind.W_LONELY,
}


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
    OPEN = "OPEN"
    SHUT = "SHUT"
    MOVE = "MOVE"
    SHIFT = "SHIFT"
    PULL = "PULL"
    SWAP = "SWAP"
    TELE = "TELE"
    FLOAT = "FLOAT"
    WEAK = "WEAK"
    MORE = "MORE"
    SAFE = "SAFE"


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
    WordKind.W_OPEN: Property.OPEN,
    WordKind.W_SHUT: Property.SHUT,
    WordKind.W_MOVE: Property.MOVE,
    WordKind.W_SHIFT: Property.SHIFT,
    WordKind.W_PULL: Property.PULL,
    WordKind.W_SWAP: Property.SWAP,
    WordKind.W_TELE: Property.TELE,
    WordKind.W_FLOAT: Property.FLOAT,
    WordKind.W_WEAK: Property.WEAK,
    WordKind.W_MORE: Property.MORE,
    WordKind.W_SAFE: Property.SAFE,
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
                EntityKind.WATER: "~~~",
                EntityKind.GRASS: "'''",
                EntityKind.TILE: "___",
                EntityKind.FLOWER: " * ",
                EntityKind.ICE: " i ",
                EntityKind.JELLY: " j ",
                EntityKind.CRAB: " c ",
                EntityKind.LOVE: " <3",
                EntityKind.ALGAE: " a ",
                EntityKind.HEDGE: " H ",
                EntityKind.BELT: " = ",
                EntityKind.BUG: " g ",
                EntityKind.ROBOT: " Rb",
                EntityKind.STAR: " * ",
            }.get(e, " ? ")
        return " . "
