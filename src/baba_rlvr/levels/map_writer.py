"""Helper to author baba-is-auto .txt maps from a readable Python spec.

baba-is-auto map format
-----------------------
First line: ``W H``. Then H rows of W space-separated integer ObjectType IDs.
Empty cells are ``ICON_EMPTY = 132``. Text tokens use the noun/verb/property
enum value (e.g. ``BABA = 4``). Icon entities use the ICON_* enum value
(e.g. ``ICON_BABA = 114``).

Rather than memorize integer IDs, level authors write a grid of short
strings (``"BABA"``, ``"is"``, ``"YOU"``, ``"baba"``, ``"."``) and we
translate via :func:`tokenize`.

Convention
----------
* UPPERCASE tokens  -> text (rule words)
* lowercase tokens  -> icon entities
* ``"."`` or empty  -> ICON_EMPTY
"""

from __future__ import annotations

from pathlib import Path

import pyBaba

# ----------------------------------------------------------------------------
# Token table. Extend as new sprites are needed.
# ----------------------------------------------------------------------------
_ICON_NAMES = [
    "baba", "rock", "wall", "flag", "skull", "lava", "key", "door", "keke",
    "water", "grass", "tile", "flower", "ice", "jelly", "crab", "love",
    "algae", "hedge", "belt", "bug", "robot", "star",
]
_TEXT_NAMES = [
    "BABA", "ROCK", "WALL", "FLAG", "SKULL", "LAVA", "KEY", "DOOR", "KEKE",
    "WATER", "GRASS", "TILE", "FLOWER", "ICE", "JELLY", "CRAB", "LOVE",
    "ALGAE", "HEDGE", "BELT", "BUG", "ROBOT", "STAR",
    "IS", "HAS", "MAKE", "AND", "NOT", "ON", "NEAR", "FACING", "LONELY",
    "YOU", "WIN", "STOP", "PUSH", "DEFEAT", "HOT", "MELT", "SINK",
    "OPEN", "SHUT", "MOVE", "SHIFT", "PULL", "SWAP", "TELE", "FLOAT",
    "WEAK", "MORE", "SAFE",
]


def _build_token_map() -> dict[str, int]:
    m: dict[str, int] = {".": int(pyBaba.ICON_EMPTY), "": int(pyBaba.ICON_EMPTY)}
    for name in _ICON_NAMES:
        m[name] = int(getattr(pyBaba, f"ICON_{name.upper()}"))
    for name in _TEXT_NAMES:
        m[name] = int(getattr(pyBaba, name))
    return m


_TOKEN_MAP = _build_token_map()
_ID_TO_TOKEN = {v: k for k, v in _TOKEN_MAP.items() if k}
_ID_TO_TOKEN[int(pyBaba.ICON_EMPTY)] = "."


def tokenize(token: str) -> int:
    """Translate a single map cell token to a pyBaba ObjectType integer."""
    t = token.strip()
    if t in _TOKEN_MAP:
        return _TOKEN_MAP[t]
    raise ValueError(f"Unknown map token: {token!r}")


def detokenize(value: int) -> str:
    """Translate a pyBaba ObjectType integer back to a readable map token."""
    if value in _ID_TO_TOKEN:
        return _ID_TO_TOKEN[value]
    raise ValueError(f"Unknown map object id: {value!r}")


def write_map(path: str | Path, rows: list[list[str]]) -> Path:
    """Write a baba-is-auto .txt map from a 2D list of token strings.

    Every row must have the same width.
    """
    h = len(rows)
    if h == 0:
        raise ValueError("rows is empty")
    w = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != w:
            raise ValueError(f"row {i} has width {len(row)}, expected {w}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{w} {h}"]
    for row in rows:
        lines.append(" ".join(str(tokenize(c)) for c in row))
    out.write_text("\n".join(lines) + "\n")
    return out


def read_map(path: str | Path) -> list[list[str]]:
    """Read a baba-is-auto .txt map into a 2D readable token grid."""
    map_path = Path(path)
    parts = map_path.read_text(encoding="utf-8").split()
    if len(parts) < 2:
        raise ValueError(f"{map_path} is not a baba-is-auto map: missing header")
    try:
        w = int(parts[0])
        h = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{map_path} has an invalid map header") from exc

    cells = parts[2:]
    if len(cells) != w * h:
        raise ValueError(f"{map_path} has {len(cells)} cells, expected {w * h}")

    rows: list[list[str]] = []
    for y in range(h):
        row: list[str] = []
        for raw in cells[y * w : (y + 1) * w]:
            try:
                row.append(detokenize(int(raw)))
            except ValueError as exc:
                raise ValueError(f"{map_path} contains unsupported cell id {raw!r}") from exc
        rows.append(row)
    return rows


def parse_grid(text: str) -> list[list[str]]:
    """Parse a triple-quoted ASCII grid into a 2D list of tokens.

    Whitespace-separated tokens within each non-empty line. Blank lines are
    ignored. Convenient for inline level definition.
    """
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(s.split())
    return rows
