import random

from baba_rlvr.engine import parse_level
from baba_rlvr.levels.loader import LEVEL_REGISTRY
from baba_rlvr.levels.map_writer import read_map
from baba_rlvr.pcg.map_elites import _mutate_spec, run_map_elites


def test_mutate_spec_materializes_map_path(tmp_path):
    spec = LEVEL_REGISTRY["tutorial_01"]

    child = _mutate_spec(spec, random.Random(0), level_id="child", out_dir=tmp_path)

    assert set(child) == {"map_path", "max_steps"}
    assert child["map_path"].endswith("child.txt")
    assert len(read_map(child["map_path"])) == len(read_map(spec["map_path"]))
    world = parse_level(child)
    assert world.width > 0


def test_run_map_elites_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr("baba_rlvr.pcg.map_elites._GENERATED_DIR", tmp_path)

    archive = run_map_elites(
        seed_levels=["tutorial_01"],
        iterations=50,
        max_solver_depth=8,
        max_solver_nodes=1_000,
        rng_seed=0,
    )

    assert archive
    assert all(set(e.spec) == {"map_path", "max_steps"} for e in archive.values())
