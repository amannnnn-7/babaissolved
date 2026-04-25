from baba_rlvr.engine import Direction, parse_level
from baba_rlvr.levels.loader import LEVEL_REGISTRY


def test_tutorial_loads():
    assert "tutorial_01" in LEVEL_REGISTRY


def test_tutorial_solvable_by_hand():
    """Walk Baba right onto the flag."""
    spec = LEVEL_REGISTRY["tutorial_01"]
    world = parse_level(spec)
    assert not world.won
    # Find Baba's row & flag column to compute a path.
    by, bx = None, None
    fy, fx = None, None
    for y, row in enumerate(world.grid):
        for x, tile in enumerate(row):
            if any(e.value == "baba" for e in tile.entities):
                by, bx = y, x
            if any(e.value == "flag" for e in tile.entities):
                fy, fx = y, x
    assert by == fy, "Tutorial level expects baba & flag on same row"
    for _ in range(fx - bx):
        info = world.step(Direction.RIGHT)
        assert not info.get("died", False)
    assert world.won, f"Expected win after walking onto flag, rules={world.rules}"


def test_rule_parsing():
    spec = LEVEL_REGISTRY["tutorial_01"]
    world = parse_level(spec)
    rule_strs = {(e.value, p.value) for e, p in world.rules}
    assert ("baba", "YOU") in rule_strs
    assert ("flag", "WIN") in rule_strs
