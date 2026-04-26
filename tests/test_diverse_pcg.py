from collections import Counter

from baba_rlvr.pcg.diverse import generate_diverse_levels


def test_generate_diverse_levels_are_unique_and_multi_family(tmp_path):
    levels = generate_diverse_levels(count=15, out_dir=tmp_path, prefix="test_diverse")

    assert len(levels) == 15
    assert len({level.spec["map_path"] for level in levels}) == 15
    assert len(Counter(level.family for level in levels)) >= 5
    assert all(level.solution for level in levels)
    assert all((tmp_path / f"{level.level_id}.txt").exists() for level in levels)
