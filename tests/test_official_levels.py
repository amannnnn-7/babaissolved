"""Tests for the official 24-level pack and tier curriculum."""

from __future__ import annotations

from pathlib import Path

import pytest

from baba_rlvr.engine import parse_level
from baba_rlvr.levels.loader import LEVEL_REGISTRY
from baba_rlvr.pcg.solver import bfs_solve
from baba_rlvr.training.curriculum import TIERS, split

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "levels" / "official"


def test_official_dir_has_all_24_levels() -> None:
    assert OFFICIAL_DIR.exists(), "Run scripts/build_official_levels.py first."
    files = sorted(p.stem for p in OFFICIAL_DIR.glob("*.txt"))
    expected = sorted(lid for lvls in TIERS.values() for lid in lvls)
    assert files == expected, f"missing or extra levels: {set(files) ^ set(expected)}"


def test_curriculum_split_holds_out_one_per_tier() -> None:
    sp = split()
    assert len(sp.train) == 16 and len(sp.eval) == 8
    for t, lvls in TIERS.items():
        eval_in_tier = [lid for lid in lvls if lid in sp.eval]
        train_in_tier = [lid for lid in lvls if lid in sp.train]
        assert len(eval_in_tier) == 1, f"tier {t} should hold out exactly 1 level"
        assert len(train_in_tier) == 2, f"tier {t} should have 2 train levels"


@pytest.mark.parametrize("lid", sorted(lid for lvls in TIERS.values() for lid in lvls))
def test_official_level_loads_and_is_solvable(lid: str) -> None:
    assert lid in LEVEL_REGISTRY, f"{lid} not registered by the loader"
    world = parse_level(LEVEL_REGISTRY[lid])
    sol = bfs_solve(world, max_depth=40, max_nodes=200_000)
    assert sol is not None, f"{lid} is unsolvable within depth 40"
    assert len(sol) <= 30, f"{lid} BFS depth={len(sol)} is too long for the curriculum"
