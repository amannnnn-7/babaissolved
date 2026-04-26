"""Curated-diverse PCG levels for the hackathon curriculum.

The MAP-Elites mutator is useful for local perturbations, but judge-facing
curriculum needs visibly different mechanics. This module emits deterministic,
solver-checked template families that exercise early Baba mechanics without
depending on brittle random edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from ..engine import parse_level
from ..levels.map_writer import write_map
from .solver import bfs_solve

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATED_DIR = _REPO_ROOT / "levels" / "_generated"


@dataclass(frozen=True)
class GeneratedLevel:
    level_id: str
    spec: dict
    solution: list[str]
    family: str


def generate_diverse_levels(
    *,
    count: int = 50,
    out_dir: Path = _GENERATED_DIR,
    prefix: str = "diverse",
    max_solver_depth: int = 50,
    max_solver_nodes: int = 30_000,
) -> list[GeneratedLevel]:
    """Materialize at least ``count`` unique, solver-checked levels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[GeneratedLevel] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()

    for family, rows in _candidate_levels():
        key = tuple(tuple(row) for row in rows)
        if key in seen:
            continue
        seen.add(key)
        level_id = f"{prefix}_{len(generated):03d}_{family}"
        path = write_map(out_dir / f"{level_id}.txt", rows).resolve()
        spec = {"map_path": str(path), "max_steps": 80}
        world = parse_level(spec)
        sol = bfs_solve(world, max_depth=max_solver_depth, max_nodes=max_solver_nodes)
        if sol is None:
            path.unlink(missing_ok=True)
            continue
        generated.append(
            GeneratedLevel(
                level_id=level_id,
                spec=spec,
                solution=[step.value for step in sol],
                family=family,
            )
        )
        if len(generated) >= count:
            return generated

    raise RuntimeError(f"Only generated {len(generated)} solvable unique levels; wanted {count}")


def _candidate_levels():
    players = [
        ("BABA", "baba"),
        ("KEKE", "keke"),
        ("CRAB", "crab"),
    ]
    openers = [
        ("KEY", "key"),
        ("LOVE", "love"),
        ("ROCK", "rock"),
    ]
    blockers = [
        ("DOOR", "door"),
        ("JELLY", "jelly"),
    ]

    families = [
        [
            (
                "open_shut",
                _open_shut_level(
                    player_word,
                    player_icon,
                    opener_word,
                    opener_icon,
                    blocker_word,
                    blocker_icon,
                ),
            )
            for player_word, player_icon in players
            for opener_word, opener_icon in openers
            for blocker_word, blocker_icon in blockers
        ],
        [
            ("water_detour", _water_detour_level(player_word, player_icon, offset))
            for player_word, player_icon in players
            for offset in range(12)
        ],
        [
            (
                "push_corridor",
                _push_corridor_level(player_word, player_icon, pusher_word, pusher_icon, gap),
            )
            for player_word, player_icon in players
            for pusher_word, pusher_icon in openers
            for gap in range(4)
        ],
        [
            ("and_stop_gate", _and_stop_gate_level(player_word, player_icon, variant))
            for player_word, player_icon in players
            for variant in range(12)
        ],
        [
            ("mixed_hazards", _mixed_hazards_level(player_word, player_icon, variant))
            for player_word, player_icon in players
            for variant in range(12)
        ],
    ]

    for batch in zip_longest(*families):
        for item in batch:
            if item is not None:
                yield item


def _row(width: int, cells: list[str]) -> list[str]:
    if len(cells) != width - 2:
        raise ValueError(f"inner row has {len(cells)} cells, expected {width - 2}")
    return ["wall", *cells, "wall"]


def _open_shut_level(
    player_word: str,
    player_icon: str,
    opener_word: str,
    opener_icon: str,
    blocker_word: str,
    blocker_icon: str,
) -> list[list[str]]:
    width = 9
    return [
        ["wall"] * width,
        _row(width, [player_word, "IS", "YOU", ".", ".", ".", "."]),
        _row(width, [opener_word, "IS", "PUSH", "AND", "OPEN", ".", "."]),
        _row(width, [blocker_word, "IS", "SHUT", ".", ".", ".", "."]),
        _row(width, ["HEDGE", "AND", blocker_word, "IS", "STOP", ".", "."]),
        _row(width, [player_icon, opener_icon, blocker_icon, "flag", ".", ".", "."]),
        _row(width, ["FLAG", "IS", "WIN", ".", "hedge", ".", "."]),
        ["wall"] * width,
    ]


def _water_detour_level(player_word: str, player_icon: str, offset: int) -> list[list[str]]:
    width = 10
    lower_water = "water" if offset % 2 else "."
    upper_water = "." if offset % 2 else "water"
    decor = ["ice", "grass", "flower"][offset % 3]
    return [
        ["wall"] * width,
        _row(width, [player_word, "IS", "YOU", ".", "WATER", "IS", "SINK", "."]),
        _row(width, ["FLAG", "IS", "WIN", ".", "ROCK", "IS", "PUSH", "."]),
        _row(width, [player_icon, ".", upper_water, ".", ".", "water", "flag", "."]),
        _row(width, [".", "wall", "wall", ".", "wall", ".", "wall", "."]),
        _row(width, [".", ".", ".", ".", ".", ".", ".", lower_water]),
        _row(width, [decor, ".", "rock", ".", "water", ".", ".", "."]),
        ["wall"] * width,
    ]


def _push_corridor_level(
    player_word: str,
    player_icon: str,
    pusher_word: str,
    pusher_icon: str,
    gap: int,
) -> list[list[str]]:
    width = 11
    spacer = "." if gap % 2 == 0 else "tile"
    return [
        ["wall"] * width,
        _row(width, [player_word, "IS", "YOU", ".", "FLAG", "IS", "WIN", ".", "."]),
        _row(width, [pusher_word, "IS", "PUSH", ".", "WALL", "IS", "STOP", ".", "."]),
        _row(width, [player_icon, pusher_icon, ".", ".", ".", "wall", "flag", ".", "."]),
        _row(width, [".", "wall", "wall", spacer, ".", "wall", ".", "wall", "."]),
        _row(width, [".", ".", ".", ".", ".", ".", ".", ".", "."]),
        ["wall"] * width,
    ]


def _and_stop_gate_level(player_word: str, player_icon: str, variant: int) -> list[list[str]]:
    width = 10
    side_obj = ["algae", "flower", "ice"][variant % 3]
    opener_word, opener_icon = [("KEY", "key"), ("LOVE", "love")][variant % 2]
    return [
        ["wall"] * width,
        _row(width, [player_word, "IS", "YOU", ".", "FLAG", "IS", "WIN", "."]),
        _row(width, [opener_word, "IS", "PUSH", "AND", "OPEN", ".", ".", "."]),
        _row(width, ["DOOR", "IS", "SHUT", ".", "HEDGE", "AND", "DOOR", "IS"]),
        _row(width, ["STOP", ".", ".", ".", ".", ".", ".", "."]),
        _row(width, [player_icon, opener_icon, "door", "flag", ".", side_obj, ".", "."]),
        ["wall"] * width,
    ]


def _mixed_hazards_level(player_word: str, player_icon: str, variant: int) -> list[list[str]]:
    width = 11
    opener_word, opener_icon = [("KEY", "key"), ("ROCK", "rock"), ("LOVE", "love")][variant % 3]
    blocker_icon = ["door", "jelly"][variant % 2]
    blocker_word = blocker_icon.upper()
    decor = ["crab", "algae", "ice", "flower"][variant % 4]
    return [
        ["wall"] * width,
        _row(width, [player_word, "IS", "YOU", ".", "WATER", "IS", "SINK", ".", "."]),
        _row(width, [opener_word, "IS", "PUSH", "AND", "OPEN", ".", ".", ".", "."]),
        _row(width, [blocker_word, "IS", "SHUT", ".", "FLAG", "IS", "WIN", ".", "."]),
        _row(width, ["HEDGE", "AND", blocker_word, "IS", "STOP", ".", ".", ".", "."]),
        _row(
            width,
            [player_icon, opener_icon, blocker_icon, "flag", ".", "water", ".", decor, "."],
        ),
        _row(width, [".", ".", ".", ".", ".", ".", ".", ".", "water"]),
        ["wall"] * width,
    ]
