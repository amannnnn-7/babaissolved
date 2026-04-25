"""Breadth-first solver over engine states.

Used both:
  * by the PCG generator to verify that candidate levels are actually solvable
    (and to estimate solution depth), and
  * by the eval harness to compute an *optimality ratio* of the trained agent.

The state hash is a tuple-of-tuples-of-tuples, making it cheap to put into a
visited set. We bound search depth and node count to keep generation tractable.
"""

from __future__ import annotations

from collections import deque

from ..engine import Direction, World

ACTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


def state_key(world: World) -> tuple:
    """Hashable canonical form of a world. step_count is excluded so the
    solver explores the *space* of board configurations, not move counts."""
    return tuple(
        tuple(
            (tile.entities, tile.words) for tile in row
        )
        for row in world.grid
    )


def bfs_solve(
    world: World, max_depth: int = 30, max_nodes: int = 50_000
) -> list[Direction] | None:
    """Return a shortest action sequence that wins, or None if not found.

    `max_depth` caps move-count, `max_nodes` caps memory. Both are intentional
    knobs the PCG loop tunes for difficulty stratification.
    """
    if world.won:
        return []
    if world.lost:
        return None

    start = world.clone()
    start.parse_rules()
    start_key = state_key(start)
    visited: set[tuple] = {start_key}
    # Each frontier entry: (world, path_so_far)
    queue: deque[tuple[World, list[Direction]]] = deque([(start, [])])
    nodes = 0

    while queue and nodes < max_nodes:
        w, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for a in ACTIONS:
            child = w.clone()
            child.step(a)
            nodes += 1
            if child.won:
                return path + [a]
            if child.lost:
                continue
            key = state_key(child)
            if key in visited:
                continue
            visited.add(key)
            queue.append((child, path + [a]))
    return None
