from baba_rlvr.engine import parse_level
from baba_rlvr.levels.loader import LEVEL_REGISTRY
from baba_rlvr.pcg.solver import bfs_solve


def test_solver_finds_tutorial():
    world = parse_level(LEVEL_REGISTRY["tutorial_01"])
    sol = bfs_solve(world, max_depth=15)
    assert sol is not None
    assert len(sol) <= 15
