from pathlib import Path

from baba_rlvr.engine import Direction
from baba_rlvr.levels.loader import load_level
from baba_rlvr.viz import (
    render_trajectory_gif,
    render_trajectory_strip,
    render_world,
    rollout_actions,
)


def test_render_world_returns_image():
    w = load_level("tutorial_01")
    img = render_world(w, cell=24, title="t")
    assert img.size[0] > 0 and img.size[1] > 0
    assert img.mode == "RGB"


def test_render_trajectory_gif(tmp_path: Path):
    w = load_level("tutorial_01")
    frames = rollout_actions(w, [Direction.RIGHT] * 6)
    out = tmp_path / "traj.gif"
    p = render_trajectory_gif(frames, out, cell=20, duration_ms=100)
    assert p.exists()
    assert p.stat().st_size > 200


def test_render_trajectory_strip(tmp_path: Path):
    w = load_level("tutorial_01")
    frames = rollout_actions(w, [Direction.RIGHT] * 6)
    out = tmp_path / "strip.png"
    p = render_trajectory_strip(frames, out, cell=18, cols=4)
    assert p.exists()
    assert p.stat().st_size > 200


def test_rollout_stops_on_terminal():
    w = load_level("tutorial_01")
    # Walk far enough to win, plus extra steps; rollout should stop at win.
    frames = rollout_actions(w, [Direction.RIGHT] * 20)
    assert frames[-1][0].won
    # Confirm we didn't keep stepping past the win.
    assert sum(1 for f in frames if f[0].won) == 1
