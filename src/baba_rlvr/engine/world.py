"""World wrapper around the C++ `pyBaba` engine (utilForever/baba-is-auto).

This module is a thin Python adapter that preserves the public `World` API
expected by the rest of the codebase (server, reward tracker, renderer,
solver, tests) while delegating actual game mechanics — push-chains, rule
parsing, NOUN-IS-NOUN transformations, defeat/sink/melt resolution — to
the battle-tested C++ engine.

Why an adapter and not a direct exposure?
-----------------------------------------
The C++ engine speaks `pyBaba.ObjectType` (a 176-member flat enum). Our
contract speaks readable string enums (`EntityKind.BABA`, `Property.WIN`)
because they round-trip into JSON observations and LLM prompts. The adapter
translates one-shot per query and caches per-step.

State cloning is implemented via *action replay* from the source map file
because the upstream `Game` class does not expose a copy constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyBaba

from .types import Direction, EntityKind, Property, Tile, WordKind

# ---------------------------------------------------------------------------
# Translation tables: pyBaba.ObjectType <-> our string enums
# ---------------------------------------------------------------------------
# Every NOUN we represent in the pure-Python contract maps to two pyBaba
# members: a TEXT noun (e.g. pyBaba.BABA) and an ICON entity (pyBaba.ICON_BABA).
_ENTITY_BY_ICON: dict[int, EntityKind] = {}
_ICON_BY_ENTITY: dict[EntityKind, int] = {}
_WORDKIND_BY_TEXT: dict[int, WordKind] = {}
_TEXT_BY_WORDKIND: dict[WordKind, int] = {}


def _register_noun(name: str, ent: EntityKind, word: WordKind) -> None:
    icon = getattr(pyBaba, f"ICON_{name}", None)
    text = getattr(pyBaba, name, None)
    if icon is not None:
        _ENTITY_BY_ICON[int(icon)] = ent
        _ICON_BY_ENTITY[ent] = int(icon)
    if text is not None:
        _WORDKIND_BY_TEXT[int(text)] = word
        _TEXT_BY_WORDKIND[word] = int(text)


_register_noun("BABA", EntityKind.BABA, WordKind.W_BABA)
_register_noun("ROCK", EntityKind.ROCK, WordKind.W_ROCK)
_register_noun("WALL", EntityKind.WALL, WordKind.W_WALL)
_register_noun("FLAG", EntityKind.FLAG, WordKind.W_FLAG)
_register_noun("SKULL", EntityKind.SKULL, WordKind.W_SKULL)
_register_noun("LAVA", EntityKind.LAVA, WordKind.W_LAVA)
_register_noun("KEKE", EntityKind.KEKE, WordKind.W_KEKE)
_register_noun("DOOR", EntityKind.DOOR, WordKind.W_DOOR)
_register_noun("KEY", EntityKind.KEY, WordKind.W_KEY)

# Verb + properties we surface in our contract.
_WORDKIND_BY_TEXT[int(pyBaba.IS)] = WordKind.W_IS
_TEXT_BY_WORDKIND[WordKind.W_IS] = int(pyBaba.IS)

_PROPERTY_BY_TEXT: dict[int, Property] = {
    int(pyBaba.YOU): Property.YOU,
    int(pyBaba.WIN): Property.WIN,
    int(pyBaba.STOP): Property.STOP,
    int(pyBaba.PUSH): Property.PUSH,
    int(pyBaba.DEFEAT): Property.DEFEAT,
    int(pyBaba.HOT): Property.HOT,
    int(pyBaba.MELT): Property.MELT,
    int(pyBaba.SINK): Property.SINK,
}
for _ot, _prop in _PROPERTY_BY_TEXT.items():
    # Map property words to our WordKind so the renderer can color them.
    _word_name = f"W_{_prop.value}"
    _wk = getattr(WordKind, _word_name, None)
    if _wk is not None:
        _WORDKIND_BY_TEXT[_ot] = _wk
        _TEXT_BY_WORDKIND[_wk] = _ot

_DIR_TO_PYBABA = {
    Direction.UP: pyBaba.Direction.UP,
    Direction.DOWN: pyBaba.Direction.DOWN,
    Direction.LEFT: pyBaba.Direction.LEFT,
    Direction.RIGHT: pyBaba.Direction.RIGHT,
    Direction.WAIT: pyBaba.Direction.NONE,
}


# ---------------------------------------------------------------------------
# Object grouping helpers (used by renderer + grid serialization)
# ---------------------------------------------------------------------------
def _classify(types: list[pyBaba.ObjectType]) -> tuple[list[EntityKind], list[WordKind]]:
    """Split a tile's list of pyBaba ObjectTypes into our (entities, words)."""
    ents: list[EntityKind] = []
    words: list[WordKind] = []
    for t in types:
        ti = int(t)
        if ti in _ENTITY_BY_ICON:
            ents.append(_ENTITY_BY_ICON[ti])
        elif ti in _WORDKIND_BY_TEXT:
            words.append(_WORDKIND_BY_TEXT[ti])
        # Unknown types (ICON_EMPTY, exotic icons we don't render) are skipped.
    return ents, words


# ---------------------------------------------------------------------------
# World adapter
# ---------------------------------------------------------------------------
@dataclass
class World:
    """Adapter over a `pyBaba.Game`.

    Source-of-truth is the underlying C++ Game; Python-side fields are
    derived caches refreshed on every step / load.
    """

    map_path: str
    height: int = 0
    width: int = 0
    max_steps: int = 80
    step_count: int = 0
    won: bool = False
    lost: bool = False
    grid: list[list[Tile]] = field(default_factory=list)
    rules: set[tuple[EntityKind, Property]] = field(default_factory=set)
    # Trajectory of actions applied since the last load(): used for clone().
    _history: list[Direction] = field(default_factory=list)
    _game: pyBaba.Game = field(init=False, repr=False)

    # ------------------------------------------------------------------ ctor
    def __post_init__(self) -> None:
        self._game = pyBaba.Game(self.map_path)
        self._refresh()

    # ---------------------------------------------------------- core helpers
    def _refresh(self) -> None:
        m = self._game.GetMap()
        self.width = int(m.GetWidth())
        self.height = int(m.GetHeight())
        # Build grid of Tile objects.
        grid: list[list[Tile]] = []
        for y in range(self.height):
            row: list[Tile] = []
            for x in range(self.width):
                ts = m.At(x, y).GetTypes()
                ents, words = _classify(list(ts))
                row.append(Tile(entities=tuple(ents), words=tuple(words)))
            grid.append(row)
        self.grid = grid
        self.rules = self._extract_rules()
        ps = self._game.GetPlayState()
        self.won = ps == pyBaba.PlayState.WON
        self.lost = ps == pyBaba.PlayState.LOST

    def _extract_rules(self) -> set[tuple[EntityKind, Property]]:
        """Read the active rule set from the C++ RuleManager.

        We surface only NOUN-IS-PROPERTY rules whose noun + property are in
        our string-enum vocabulary. Exotic rules still execute in C++; they
        just don't show up in the verifier-visible projection.
        """
        rm = self._game.GetRuleManager()
        seen: set[tuple[int, int]] = set()
        out: set[tuple[EntityKind, Property]] = set()
        # Iterate over every noun token we know about and collect its rules.
        for ent, icon_id in _ICON_BY_ENTITY.items():
            text_id = _TEXT_BY_WORDKIND.get(_word_for_entity(ent))
            if text_id is None:
                continue
            try:
                rules = rm.GetRules(pyBaba.ObjectType(text_id))
            except Exception:
                continue
            for r in rules:
                o1, _o2, o3 = r.objects
                t1 = _first_type(o1)
                t3 = _first_type(o3)
                if t1 is None or t3 is None:
                    continue
                key = (int(t1), int(t3))
                if key in seen:
                    continue
                seen.add(key)
                noun_kind = _WORDKIND_BY_TEXT.get(int(t1))
                if noun_kind is None or noun_kind not in _NOUN_TO_ENTITY:
                    continue
                prop = _PROPERTY_BY_TEXT.get(int(t3))
                if prop is None:
                    continue
                out.add((_NOUN_TO_ENTITY[noun_kind], prop))
        return out

    # ------------------------------------------------------------ public API
    def step(self, direction: Direction) -> dict:
        """Advance the world. Returns an info dict matching the legacy API."""
        info: dict = {"died": False, "moved": False, "invalid_move": False}
        if self.won or self.lost:
            return info
        prev_state = (self.won, self.lost)
        prev_you_count = sum(
            1
            for row in self.grid
            for tile in row
            for e in tile.entities
            if e in self.you_entities()
        )
        self._game.MovePlayer(_DIR_TO_PYBABA[direction])
        self._history.append(direction)
        self.step_count += 1
        self._refresh()
        # Movement detection: compare YOU positions before/after.
        new_you = self.you_entities()
        new_you_count = sum(
            1
            for row in self.grid
            for tile in row
            for e in tile.entities
            if e in new_you
        )
        # Heuristic: if a YOU exists and the play state didn't change but no
        # observable transition happened, treat it as a no-op. We can't easily
        # tell from pyBaba whether the move was legal; we use the conservative
        # rule that losing a YOU on this step counts as a death.
        if new_you_count < prev_you_count:
            info["died"] = True
        info["moved"] = direction != Direction.WAIT
        if self.step_count >= self.max_steps:
            info["truncated"] = True
        return info

    def clone(self) -> World:
        """Return an independent copy by replaying the action history."""
        new = World(map_path=self.map_path, max_steps=self.max_steps)
        for a in self._history:
            new._game.MovePlayer(_DIR_TO_PYBABA[a])
        new._history = list(self._history)
        new.step_count = self.step_count
        new._refresh()
        return new

    def parse_rules(self) -> set[tuple[EntityKind, Property]]:
        """Refresh + return active rules (kept for legacy callers)."""
        self.rules = self._extract_rules()
        return self.rules

    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def tile(self, y: int, x: int) -> Tile:
        return self.grid[y][x]

    def you_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p == Property.YOU}

    def win_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p == Property.WIN}

    def stop_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p == Property.STOP}

    def push_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p == Property.PUSH}

    def kill_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p == Property.DEFEAT}

    # ------------------------------------------------------- serialization
    def render_ascii(self) -> str:
        rows = []
        for row in self.grid:
            rows.append("|".join(t.render() for t in row))
        return "\n".join(rows)

    def to_tokens(self) -> list[list[str]]:
        out: list[list[str]] = []
        for row in self.grid:
            out_row: list[str] = []
            for t in row:
                parts = [w.value for w in t.words] + [e.value for e in t.entities]
                out_row.append(",".join(parts) if parts else ".")
            out.append(out_row)
        return out


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------
_NOUN_TO_ENTITY: dict[WordKind, EntityKind] = {
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

_ENTITY_TO_NOUN: dict[EntityKind, WordKind] = {v: k for k, v in _NOUN_TO_ENTITY.items()}


def _word_for_entity(ent: EntityKind) -> WordKind:
    return _ENTITY_TO_NOUN[ent]


def _first_type(obj: pyBaba.Object) -> pyBaba.ObjectType | None:
    ts = obj.GetTypes()
    return ts[0] if ts else None


# ---------------------------------------------------------------------------
# Level loading
# ---------------------------------------------------------------------------
def parse_level(spec: dict) -> World:
    """Load a level from a spec dict.

    Spec format:
      {"map_path": "/abs/path/to/map.txt", "max_steps": 80}
    """
    if "map_path" not in spec:
        raise ValueError("level spec must contain 'map_path'")
    return World(map_path=str(spec["map_path"]), max_steps=int(spec.get("max_steps", 80)))
