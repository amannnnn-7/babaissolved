"""World state and the step() function for Baba Is You.

Implementation notes
--------------------
The standard Baba step ordering is:
  1. Parse the rule set from the current text-block layout.
  2. Move every YOU-entity in the chosen direction.
     * Pushing PUSH/text blocks chains.
     * STOP entities block movement.
  3. Re-parse rules (movement may have changed them).
  4. Apply interactions: KILL/DEFEAT/HOT+MELT/SINK/WIN.
  5. Check terminal conditions:
        - won  : any YOU entity overlaps a WIN entity.
        - lost : no YOU entity exists (all dead, or rule removed).

We keep the implementation small and clear; correctness is verified by
tests in tests/test_engine.py.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .types import (
    NOUN_WORDS,
    PROPERTY_WORDS,
    VERB_WORDS,
    Direction,
    EntityKind,
    Property,
    Tile,
    WordKind,
)

# A rule is (subject_entity, property) e.g. (BABA, YOU). We also support
# noun-IS-noun *transformations* but defer those for v0 (TODO post-hackathon).
Rule = tuple[EntityKind, Property]


@dataclass
class World:
    height: int
    width: int
    # grid[y][x] -> Tile
    grid: list[list[Tile]]
    step_count: int = 0
    max_steps: int = 80
    won: bool = False
    lost: bool = False
    # Cached for the verifier; refreshed every step.
    rules: set[Rule] = field(default_factory=set)

    # ------------------------------------------------------------------ utils
    def clone(self) -> World:
        return deepcopy(self)

    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def tile(self, y: int, x: int) -> Tile:
        return self.grid[y][x]

    def set_tile(self, y: int, x: int, t: Tile) -> None:
        self.grid[y][x] = t

    # --------------------------------------------------------- rule parsing
    def parse_rules(self) -> set[Rule]:
        rules: set[Rule] = set()
        # Horizontal scans
        for y in range(self.height):
            for x in range(self.width - 2):
                rules |= self._triple(y, x, dy=0, dx=1)
        # Vertical scans
        for y in range(self.height - 2):
            for x in range(self.width):
                rules |= self._triple(y, x, dy=1, dx=0)
        self.rules = rules
        return rules

    def _triple(self, y: int, x: int, *, dy: int, dx: int) -> set[Rule]:
        a = self.tile(y, x).words
        b = self.tile(y + dy, x + dx).words
        c = self.tile(y + 2 * dy, x + 2 * dx).words
        if not (a and b and c):
            return set()
        if b[0] not in VERB_WORDS:
            return set()
        if a[0] not in NOUN_WORDS:
            return set()
        if c[0] not in PROPERTY_WORDS:
            # NOUN-IS-NOUN transformations not implemented in v0
            return set()
        return {(NOUN_WORDS[a[0]], PROPERTY_WORDS[c[0]])}

    # --------------------------------------------------------- queries
    def you_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p == Property.YOU}

    def win_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p == Property.WIN}

    def stop_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p == Property.STOP}

    def push_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p == Property.PUSH}

    def kill_entities(self) -> set[EntityKind]:
        return {e for e, p in self.rules if p in (Property.KILL, Property.DEFEAT)}

    # --------------------------------------------------------- step
    def step(self, direction: Direction) -> dict:
        """Advance the world by one tick. Returns an info dict."""
        self.parse_rules()
        info: dict = {"died": False, "moved": False, "invalid_move": False}
        if self.won or self.lost:
            return info

        self.step_count += 1
        if direction != Direction.WAIT:
            moved = self._move_you(direction)
            info["moved"] = moved
            info["invalid_move"] = not moved

        # Re-parse: pushing words around may have changed the rules.
        self.parse_rules()

        died = self._apply_destructions()
        info["died"] = died

        # Win check: any YOU entity overlaps a WIN entity.
        you = self.you_entities()
        win = self.win_entities()
        if you and win:
            for y in range(self.height):
                for x in range(self.width):
                    ents = self.grid[y][x].entities
                    if any(e in you for e in ents) and any(e in win for e in ents):
                        self.won = True
                        break
                if self.won:
                    break

        if not self.you_entities():
            self.lost = True

        if self.step_count >= self.max_steps:
            info["truncated"] = True

        return info

    # ------------------------ movement & pushing ----------------------------
    def _move_you(self, direction: Direction) -> bool:
        """Move every YOU entity by `direction`, pushing PUSH/text/YOU chains.

        Returns True if at least one YOU entity actually moved.
        """
        dy, dx = direction.delta
        you = self.you_entities()
        if not you:
            return False

        # Collect YOU positions; sort so we resolve the trailing edge first
        # (entities further along the move direction move first to make space).
        positions: list[tuple[int, int, EntityKind]] = []
        for y in range(self.height):
            for x in range(self.width):
                for e in self.grid[y][x].entities:
                    if e in you:
                        positions.append((y, x, e))
        positions.sort(key=lambda p: (-dy * p[0], -dx * p[1]))

        any_moved = False
        for y, x, e in positions:
            if self._try_push(y, x, e, dy, dx):
                any_moved = True
        return any_moved

    def _try_push(self, y: int, x: int, ent: EntityKind, dy: int, dx: int) -> bool:
        """Try to move a single object at (y, x) by (dy, dx).

        Returns True if it moved. Recursively pushes pushable objects in front.
        """
        ny, nx = y + dy, x + dx
        if not self.in_bounds(ny, nx):
            return False

        target = self.grid[ny][nx]
        stop = self.stop_entities()
        push = self.push_entities()

        # A YOU entity is also implicitly pushable by another YOU? In Baba: no,
        # YOU entities don't push each other; the second YOU would just stay.
        # We approximate by treating overlapping YOUs as non-blocking.

        # Anything in target that's STOP and not also pushable blocks us.
        for tent in target.entities:
            if tent in stop and tent not in push and tent not in self.you_entities():
                return False

        # Push any PUSH-entity or any text block that occupies the target tile.
        # If we can't push them, we can't move.
        if target.words:
            # Text blocks are always pushable.
            for w in target.words:
                if not self._try_push_word(ny, nx, w, dy, dx):
                    return False
        for tent in list(target.entities):
            if tent in push:
                if not self._try_push(ny, nx, tent, dy, dx):
                    return False

        # All clear (or successfully pushed): move ourselves.
        return self._move_entity(y, x, ny, nx, ent)

    def _try_push_word(self, y: int, x: int, word: WordKind, dy: int, dx: int) -> bool:
        ny, nx = y + dy, x + dx
        if not self.in_bounds(ny, nx):
            return False
        target = self.grid[ny][nx]
        stop = self.stop_entities()
        push = self.push_entities()
        for tent in target.entities:
            if tent in stop and tent not in push and tent not in self.you_entities():
                return False
        # Cascade: push any objects in the way.
        for w in target.words:
            if not self._try_push_word(ny, nx, w, dy, dx):
                return False
        for tent in list(target.entities):
            if tent in push:
                if not self._try_push(ny, nx, tent, dy, dx):
                    return False
        return self._move_word(y, x, ny, nx, word)

    def _move_entity(self, y: int, x: int, ny: int, nx: int, ent: EntityKind) -> bool:
        src = self.grid[y][x]
        new_src_ents = list(src.entities)
        new_src_ents.remove(ent)
        self.grid[y][x] = Tile(entities=tuple(new_src_ents), words=src.words)
        dst = self.grid[ny][nx]
        self.grid[ny][nx] = Tile(entities=dst.entities + (ent,), words=dst.words)
        return True

    def _move_word(self, y: int, x: int, ny: int, nx: int, word: WordKind) -> bool:
        src = self.grid[y][x]
        new_src_words = list(src.words)
        new_src_words.remove(word)
        self.grid[y][x] = Tile(entities=src.entities, words=tuple(new_src_words))
        dst = self.grid[ny][nx]
        self.grid[ny][nx] = Tile(entities=dst.entities, words=dst.words + (word,))
        return True

    # ------------------------ destructions ---------------------------------
    def _apply_destructions(self) -> bool:
        """Apply KILL/DEFEAT/HOT+MELT/SINK; return True if any YOU died."""
        kill = self.kill_entities()
        sink = {e for e, p in self.rules if p == Property.SINK}
        hot = {e for e, p in self.rules if p == Property.HOT}
        melt = {e for e, p in self.rules if p == Property.MELT}
        you = self.you_entities()
        any_you_died = False

        for y in range(self.height):
            for x in range(self.width):
                ents = list(self.grid[y][x].entities)
                if len(ents) < 2:
                    continue
                # KILL/DEFEAT: any YOU sharing a tile with a KILL entity dies.
                if any(e in kill for e in ents) and any(e in you for e in ents):
                    new_ents = [e for e in ents if e not in you]
                    if len(new_ents) != len(ents):
                        any_you_died = True
                    ents = new_ents
                # SINK: any two entities on a SINK tile both vanish.
                if any(e in sink for e in ents) and len(ents) >= 2:
                    if any(e in you for e in ents):
                        any_you_died = True
                    ents = []
                # HOT + MELT: melt entities die on hot tiles.
                if any(e in hot for e in ents) and any(e in melt for e in ents):
                    new_ents = [e for e in ents if e not in melt]
                    if any(e in you for e in ents if e in melt):
                        any_you_died = True
                    ents = new_ents
                self.grid[y][x] = Tile(entities=tuple(ents), words=self.grid[y][x].words)
        return any_you_died

    # ------------------------ rendering / serialization --------------------
    def render_ascii(self) -> str:
        rows = []
        for row in self.grid:
            rows.append("|".join(t.render() for t in row))
        return "\n".join(rows)

    def to_tokens(self) -> list[list[str]]:
        out = []
        for row in self.grid:
            out_row = []
            for t in row:
                parts = [w.value for w in t.words] + [e.value for e in t.entities]
                out_row.append(",".join(parts) if parts else ".")
            out.append(out_row)
        return out


# ----------------------------------------------------------------------------
# Level loading
# ----------------------------------------------------------------------------
# Level format: a YAML / dict with `width`, `height`, `max_steps`, and `tiles`
# where `tiles` is a list of (y, x, kind, value) entries. We also support an
# ASCII shorthand for handcrafted levels — see levels/templates/.
#
# ASCII shorthand legend:
#   '.' empty           '#' wall              'b' baba (entity)
#   'r' rock            'F' flag              'S' skull
#   'L' lava            'D' door              'K' key   'k' keke
#   Words are written in CAPS surrounded by brackets, e.g. [BABA] [IS] [YOU].
#   Any other token is ignored.

ENTITY_GLYPHS: dict[str, EntityKind] = {
    "b": EntityKind.BABA,
    "r": EntityKind.ROCK,
    "#": EntityKind.WALL,
    "F": EntityKind.FLAG,
    "S": EntityKind.SKULL,
    "L": EntityKind.LAVA,
    "D": EntityKind.DOOR,
    "K": EntityKind.KEY,
    "k": EntityKind.KEKE,
}

WORD_NAMES: dict[str, WordKind] = {w.value: w for w in WordKind}


def parse_level(spec: dict) -> World:
    """Build a World from a structured spec.

    Spec format:
      {
        "height": int, "width": int, "max_steps": int (optional, default 80),
        "rows": ["row0 tokens space-separated", "row1 ...", ...]
      }

    Each cell token is either:
      - "." (empty)
      - one of ENTITY_GLYPHS keys (single char) or "wall" "baba" etc.
      - a WordKind value (BABA, IS, YOU, ...)
      - a comma-joined combo: e.g. "baba,WALL"  (multi-occupancy tile)
    """
    h, w = spec["height"], spec["width"]
    max_steps = spec.get("max_steps", 80)
    rows = spec["rows"]
    if len(rows) != h:
        raise ValueError(f"expected {h} rows, got {len(rows)}")

    grid: list[list[Tile]] = []
    for y, row_str in enumerate(rows):
        cells = row_str.split()
        if len(cells) != w:
            raise ValueError(f"row {y}: expected {w} cells, got {len(cells)}: {cells!r}")
        out_row: list[Tile] = []
        for cell in cells:
            ents: list[EntityKind] = []
            words: list[WordKind] = []
            if cell != ".":
                for tok in cell.split(","):
                    if tok in WORD_NAMES:
                        words.append(WORD_NAMES[tok])
                    elif tok in ENTITY_GLYPHS:
                        ents.append(ENTITY_GLYPHS[tok])
                    elif tok.lower() in (e.value for e in EntityKind):
                        ents.append(EntityKind(tok.lower()))
                    else:
                        raise ValueError(f"unknown token {tok!r} at ({y},{len(out_row)})")
            out_row.append(Tile(entities=tuple(ents), words=tuple(words)))
        grid.append(out_row)

    world = World(height=h, width=w, grid=grid, max_steps=max_steps)
    world.parse_rules()
    return world
