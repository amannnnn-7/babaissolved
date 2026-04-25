"""MAP-Elites procedural generation for Baba Is You.

Behavior descriptors (cells of the archive grid):
    axis 0:  number of rule modifications used along the optimal solution
             (0 = pure pathfinding, 1+ = requires rewriting rules)
    axis 1:  bucketed solution length

The descriptor is intentionally low-dimensional so we get visible diversity
without burning compute. Each cell stores the highest-difficulty level found
so far. Difficulty is solution_length * branching_estimate.

For the hackathon we ship a *small but honest* generator: we mutate the
hand-built templates rather than synthesizing levels from scratch. This
keeps every emitted level grammatically valid while still producing
substantial mechanic diversity (different rule layouts, wall arrangements,
swapped entities, etc.).
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass

from ..engine import Direction, parse_level
from ..levels.loader import LEVEL_REGISTRY
from .solver import bfs_solve

# (rules_modified_in_solution_bucket, sol_len_bucket) -> Elite
N_RULE_BUCKETS = 4   # 0, 1, 2, 3+
N_LEN_BUCKETS = 6    # <5, 5-9, 10-14, 15-19, 20-29, 30+


@dataclass
class Elite:
    level_id: str
    spec: dict
    solution: list[str]
    rules_modified: int
    sol_len: int
    difficulty: float

    def descriptor(self) -> tuple[int, int]:
        return (
            min(self.rules_modified, N_RULE_BUCKETS - 1),
            _len_bucket(self.sol_len),
        )


def _len_bucket(n: int) -> int:
    for i, hi in enumerate([5, 10, 15, 20, 30]):
        if n < hi:
            return i
    return N_LEN_BUCKETS - 1


# --------------------------------------------------------------- mutations
def _mutate_spec(spec: dict, rng: random.Random) -> dict:
    """Apply a small random edit. Always returns a *valid* grid (same dims)."""
    out = deepcopy(spec)
    rows = [r.split() for r in out["rows"]]
    h, w = len(rows), len(rows[0])
    op = rng.choice(["swap_cells", "place_wall", "remove_wall", "shift_rule"])

    if op == "swap_cells":
        y1, x1 = rng.randrange(h), rng.randrange(w)
        y2, x2 = rng.randrange(h), rng.randrange(w)
        rows[y1][x1], rows[y2][x2] = rows[y2][x2], rows[y1][x1]
    elif op == "place_wall":
        y, x = rng.randrange(1, h - 1), rng.randrange(1, w - 1)
        if rows[y][x] == ".":
            rows[y][x] = "#"
    elif op == "remove_wall":
        ws = [(y, x) for y in range(1, h - 1) for x in range(1, w - 1) if rows[y][x] == "#"]
        if ws:
            y, x = rng.choice(ws)
            rows[y][x] = "."
    elif op == "shift_rule":
        # Find an "IS" word and try to swap its row neighbours.
        for y in range(h):
            for x in range(1, w - 1):
                if rows[y][x] == "IS" and rng.random() < 0.5:
                    rows[y][x - 1], rows[y][x + 1] = rows[y][x + 1], rows[y][x - 1]
                    break

    out["rows"] = [" ".join(r) for r in rows]
    return out


def _count_rules_modified(spec: dict, solution: list[Direction]) -> int:
    world = parse_level(spec)
    initial = set(world.rules)
    seen_changes: set[tuple] = set()
    for a in solution:
        before = set(world.rules)
        world.step(a)
        after = set(world.rules)
        diff = (before ^ after)
        for r in diff:
            seen_changes.add(r)
    # rough count of distinct rules that ever differed from initial.
    return len(seen_changes - initial) + len({r for r in initial if r not in world.rules})


def _difficulty(sol_len: int, rules_modified: int) -> float:
    return sol_len * 0.5 + rules_modified * 2.0


# --------------------------------------------------------------- driver
def run_map_elites(
    seed_levels: list[str] | None = None,
    iterations: int = 2_000,
    max_solver_depth: int = 25,
    rng_seed: int = 0,
) -> dict[tuple[int, int], Elite]:
    """Run MAP-Elites and return the archive."""
    rng = random.Random(rng_seed)
    seeds = seed_levels or list(LEVEL_REGISTRY.keys())
    archive: dict[tuple[int, int], Elite] = {}

    # Seed the archive with the hand-built levels.
    for sid in seeds:
        spec = deepcopy(LEVEL_REGISTRY[sid])
        world = parse_level(spec)
        sol = bfs_solve(world, max_depth=max_solver_depth)
        if sol is None:
            continue
        rm = _count_rules_modified(spec, sol)
        e = Elite(
            level_id=f"seed_{sid}",
            spec=spec,
            solution=[a.value for a in sol],
            rules_modified=rm,
            sol_len=len(sol),
            difficulty=_difficulty(len(sol), rm),
        )
        archive[e.descriptor()] = e

    if not archive:
        raise RuntimeError("No solvable seed levels — cannot start MAP-Elites.")

    for it in range(iterations):
        parent = rng.choice(list(archive.values()))
        child_spec = _mutate_spec(parent.spec, rng)
        try:
            world = parse_level(child_spec)
        except ValueError:
            continue
        sol = bfs_solve(world, max_depth=max_solver_depth)
        if sol is None:
            continue
        rm = _count_rules_modified(child_spec, sol)
        diff = _difficulty(len(sol), rm)
        e = Elite(
            level_id=f"pcg_{it:05d}",
            spec=child_spec,
            solution=[a.value for a in sol],
            rules_modified=rm,
            sol_len=len(sol),
            difficulty=diff,
        )
        cell = e.descriptor()
        if cell not in archive or diff > archive[cell].difficulty:
            archive[cell] = e
    return archive
