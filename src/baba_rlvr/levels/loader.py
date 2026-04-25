"""Level loader.

Levels can come from three sources:
    1. Built-in YAML files under levels/templates/*.yaml (loaded at import).
    2. Programmatic registration via register_level(level_id, spec_dict).
    3. PCG-generated archives loaded via levels/_generated/<id>.yaml.

A "spec" is the dict consumed by engine.parse_level().
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from ..engine import World, parse_level

LEVEL_REGISTRY: dict[str, dict] = {}

# templates/ ships next to this file *and* under repo-root/levels/templates/.
# We support both for convenience (installed package vs editable dev).
_HERE = Path(__file__).parent
# loader.py lives at src/baba_rlvr/levels/loader.py — three parents up is the repo root.
_REPO_ROOT_LEVELS = _HERE.parent.parent.parent / "levels" / "templates"
_PKG_LEVELS = _HERE / "templates"


def _load_yaml_dir(path: Path) -> None:
    if not path.exists():
        return
    for f in sorted(path.glob("*.yaml")):
        spec = yaml.safe_load(f.read_text())
        if not isinstance(spec, dict) or "rows" not in spec:
            continue
        level_id = spec.get("id") or f.stem
        LEVEL_REGISTRY[level_id] = spec


_load_yaml_dir(_PKG_LEVELS)
_load_yaml_dir(_REPO_ROOT_LEVELS)


def register_level(level_id: str, spec: dict) -> None:
    LEVEL_REGISTRY[level_id] = deepcopy(spec)


def load_level(level_id: str) -> World:
    if level_id not in LEVEL_REGISTRY:
        raise KeyError(
            f"Unknown level_id={level_id!r}. Registered: {sorted(LEVEL_REGISTRY)[:5]}..."
        )
    return parse_level(deepcopy(LEVEL_REGISTRY[level_id]))
