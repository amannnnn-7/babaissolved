"""Level loader.

Levels are stored as **baba-is-auto .txt maps** (the format consumed by
`pyBaba.Game(path)`). They live in:

  * ``levels/templates/*.txt``        — handcrafted RLVR puzzles
  * ``levels/_generated/*.txt``       — MAP-Elites generated puzzles
  * ``vendor/baba-is-auto/Resources/Maps/*.txt`` — upstream demo levels
  * programmatically registered specs (e.g. PCG output via
    :func:`register_level`).

A *spec* in the registry is a dict ``{"map_path": "...", "max_steps": int}``
consumed by :func:`baba_rlvr.engine.parse_level`.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ..engine import World, parse_level

LEVEL_REGISTRY: dict[str, dict] = {}

_HERE = Path(__file__).parent
# loader.py lives at src/baba_rlvr/levels/loader.py — three parents up = repo root.
_REPO_ROOT = _HERE.parent.parent.parent
_TEMPLATES = _REPO_ROOT / "levels" / "templates"
_OFFICIAL = _REPO_ROOT / "levels" / "official"
_GENERATED = _REPO_ROOT / "levels" / "_generated"
_VENDOR_MAPS = _REPO_ROOT / "vendor" / "baba-is-auto" / "Resources" / "Maps"

_DEFAULT_MAX_STEPS: dict[str, int] = {
    "tutorial_01": 30,
    "use_mention_01": 30,
    "schema_drift_01": 40,
    "self_redefine_01": 50,
}


def _scan_dir(path: Path, *, prefix: str = "") -> None:
    if not path.exists():
        return
    for f in sorted(path.glob("*.txt")):
        # Skip files that don't parse as ``W H ...``: they're not maps.
        try:
            head = f.read_text(encoding="utf-8", errors="ignore").split(None, 2)
            int(head[0])
            int(head[1])
        except (ValueError, IndexError):
            continue
        level_id = f"{prefix}{f.stem}"
        LEVEL_REGISTRY[level_id] = {
            "map_path": str(f.resolve()),
            "max_steps": _DEFAULT_MAX_STEPS.get(f.stem, 80),
        }


_scan_dir(_TEMPLATES)
_scan_dir(_OFFICIAL)
_scan_dir(_GENERATED)
_scan_dir(_VENDOR_MAPS, prefix="vendor_")


def register_level(level_id: str, spec: dict) -> None:
    LEVEL_REGISTRY[level_id] = deepcopy(spec)


def load_level(level_id: str) -> World:
    if level_id not in LEVEL_REGISTRY:
        raise KeyError(
            f"Unknown level_id={level_id!r}. Registered: {sorted(LEVEL_REGISTRY)[:5]}..."
        )
    return parse_level(deepcopy(LEVEL_REGISTRY[level_id]))
