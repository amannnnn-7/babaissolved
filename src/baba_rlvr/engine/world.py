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
_register_noun("WATER", EntityKind.WATER, WordKind.W_WATER)
_register_noun("GRASS", EntityKind.GRASS, WordKind.W_GRASS)
_register_noun("TILE", EntityKind.TILE, WordKind.W_TILE)
_register_noun("FLOWER", EntityKind.FLOWER, WordKind.W_FLOWER)
_register_noun("ICE", EntityKind.ICE, WordKind.W_ICE)
_register_noun("JELLY", EntityKind.JELLY, WordKind.W_JELLY)
_register_noun("CRAB", EntityKind.CRAB, WordKind.W_CRAB)
_register_noun("LOVE", EntityKind.LOVE, WordKind.W_LOVE)
_register_noun("ALGAE", EntityKind.ALGAE, WordKind.W_ALGAE)
_register_noun("HEDGE", EntityKind.HEDGE, WordKind.W_HEDGE)
_register_noun("BELT", EntityKind.BELT, WordKind.W_BELT)
_register_noun("BUG", EntityKind.BUG, WordKind.W_BUG)
_register_noun("ROBOT", EntityKind.ROBOT, WordKind.W_ROBOT)
_register_noun("STAR", EntityKind.STAR, WordKind.W_STAR)

# Verb + properties we surface in our contract.
for _name, _wk in {
    "IS": WordKind.W_IS,
    "HAS": WordKind.W_HAS,
    "MAKE": WordKind.W_MAKE,
    "AND": WordKind.W_AND,
    "NOT": WordKind.W_NOT,
    "ON": WordKind.W_ON,
    "NEAR": WordKind.W_NEAR,
    "FACING": WordKind.W_FACING,
    "LONELY": WordKind.W_LONELY,
}.items():
    _obj = getattr(pyBaba, _name, None)
    if _obj is not None:
        _WORDKIND_BY_TEXT[int(_obj)] = _wk
        _TEXT_BY_WORDKIND[_wk] = int(_obj)

_PROPERTY_BY_TEXT: dict[int, Property] = {
    int(pyBaba.YOU): Property.YOU,
    int(pyBaba.WIN): Property.WIN,
    int(pyBaba.STOP): Property.STOP,
    int(pyBaba.PUSH): Property.PUSH,
    int(pyBaba.DEFEAT): Property.DEFEAT,
    int(pyBaba.HOT): Property.HOT,
    int(pyBaba.MELT): Property.MELT,
    int(pyBaba.SINK): Property.SINK,
    int(pyBaba.OPEN): Property.OPEN,
    int(pyBaba.SHUT): Property.SHUT,
    int(pyBaba.MOVE): Property.MOVE,
    int(pyBaba.SHIFT): Property.SHIFT,
    int(pyBaba.PULL): Property.PULL,
    int(pyBaba.SWAP): Property.SWAP,
    int(pyBaba.TELE): Property.TELE,
    int(pyBaba.FLOAT): Property.FLOAT,
    int(pyBaba.WEAK): Property.WEAK,
    int(pyBaba.MORE): Property.MORE,
    int(pyBaba.SAFE): Property.SAFE,
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
        self._augment_rules()
        self.rules = self._extract_rules()
        ps = self._game.GetPlayState()
        self.won = ps == pyBaba.PlayState.WON
        self.lost = ps == pyBaba.PlayState.LOST

    def _augment_rules(self) -> None:
        """Add AND-expanded rules that baba-is-auto's parser does not emit."""
        rm = self._game.GetRuleManager()
        existing = {
            _rule_key(rule)
            for known in (*_TEXT_BY_WORDKIND.values(), *_PROPERTY_BY_TEXT.keys())
            for rule in rm.GetRules(pyBaba.ObjectType(known))
        }
        to_add = [rule for rule in self._scan_and_rules() if rule not in existing]
        for subject, verb, predicate in to_add:
            key = (subject, verb, predicate)
            rm.AddRule(
                pyBaba.Rule(
                    pyBaba.Object([pyBaba.ObjectType(subject)]),
                    pyBaba.Object([pyBaba.ObjectType(verb)]),
                    pyBaba.Object([pyBaba.ObjectType(predicate)]),
                )
            )
            existing.add(key)

    def _scan_and_rules(self) -> set[tuple[int, int, int]]:
        out: set[tuple[int, int, int]] = set()
        for y in range(self.height):
            for x in range(self.width):
                out |= self._scan_line(x, y, 1, 0)
                out |= self._scan_line(x, y, 0, 1)
        return out

    def _scan_line(self, x: int, y: int, dx: int, dy: int) -> set[tuple[int, int, int]]:
        tokens: list[int] = []
        cx, cy = x, y
        while 0 <= cx < self.width and 0 <= cy < self.height:
            text = _first_text_type(self._game.GetMap().At(cx, cy))
            if text is None:
                break
            tokens.append(int(text))
            cx += dx
            cy += dy
        return _expand_rule_tokens(tokens)

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
        for ent in _ICON_BY_ENTITY:
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

_ENTITY_TO_NOUN: dict[EntityKind, WordKind] = {v: k for k, v in _NOUN_TO_ENTITY.items()}


def _word_for_entity(ent: EntityKind) -> WordKind:
    return _ENTITY_TO_NOUN[ent]


def _first_type(obj: pyBaba.Object) -> pyBaba.ObjectType | None:
    ts = obj.GetTypes()
    return ts[0] if ts else None


def _first_text_type(obj: pyBaba.Object) -> pyBaba.ObjectType | None:
    for t in obj.GetTypes():
        ti = int(t)
        if ti in _WORDKIND_BY_TEXT or ti in _PROPERTY_BY_TEXT:
            return t
    return None


def _rule_key(rule) -> tuple[int | None, int | None, int | None]:
    o1, o2, o3 = rule.objects
    t1 = _first_type(o1)
    t2 = _first_type(o2)
    t3 = _first_type(o3)
    return (
        int(t1) if t1 is not None else None,
        int(t2) if t2 is not None else None,
        int(t3) if t3 is not None else None,
    )


def _expand_rule_tokens(tokens: list[int]) -> set[tuple[int, int, int]]:
    rules: set[tuple[int, int, int]] = set()
    for verb_i, verb in enumerate(tokens):
        if verb not in _VERB_TOKENS:
            continue
        subjects = _collect_terms(tokens, verb_i - 1, -1, _NOUN_TEXT_TOKENS)
        predicates = _collect_terms(
            tokens,
            verb_i + 1,
            1,
            _NOUN_TEXT_TOKENS | set(_PROPERTY_BY_TEXT),
        )
        for subject in subjects:
            for predicate in predicates:
                rules.add((subject, verb, predicate))
    return rules


def _collect_terms(tokens: list[int], start: int, step: int, allowed: set[int]) -> list[int]:
    terms: list[int] = []
    i = start
    expect_term = True
    while 0 <= i < len(tokens):
        token = tokens[i]
        if expect_term:
            if token not in allowed:
                break
            terms.append(token)
        elif token != int(pyBaba.AND):
            break
        expect_term = not expect_term
        i += step
    if step < 0:
        terms.reverse()
    return terms if not expect_term else []


_NOUN_TEXT_TOKENS = {int(k) for k in _WORDKIND_BY_TEXT if _WORDKIND_BY_TEXT[k] in _NOUN_TO_ENTITY}
_VERB_TOKENS = {
    int(_TEXT_BY_WORDKIND[w])
    for w in (WordKind.W_IS, WordKind.W_HAS, WordKind.W_MAKE)
    if w in _TEXT_BY_WORDKIND
}


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
