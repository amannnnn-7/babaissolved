"""Author 24 progression-themed Baba Is You levels.

Eight difficulty tiers (T1..T8) × three levels each. Within each tier the
first two are *training* levels and the third (`*_eval`) is the held-out
evaluation level. The set is intentionally small, fast to BFS-verify, and
covers the early-game mechanic vocabulary supported by our pyBaba adapter:

  T1  movement only           (BABA IS YOU, FLAG IS WIN preset)
  T2  STOP walls              (WALL IS STOP gates a path)
  T3  PUSH rocks              (ROCK IS PUSH; rock blocks path)
  T4  break a STOP rule       (push WALL/STOP word aside)
  T5  SINK / water hazards    (WATER IS SINK — detour or sink a rock)
  T6  form a new WIN rule     (FLAG/ROCK + IS + WIN scattered)
  T7  schema drift / IS YOU   (change which entity IS YOU)
  T8  OPEN / SHUT (key/door)  (KEY IS OPEN destroys SHUT door)

Run::

    uv run python scripts/build_official_levels.py

Each level is written to ``levels/official/<id>.txt`` and BFS-verified.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baba_rlvr.engine import parse_level  # noqa: E402
from baba_rlvr.levels.map_writer import parse_grid, write_map  # noqa: E402
from baba_rlvr.pcg.solver import bfs_solve  # noqa: E402

OUT_DIR = ROOT / "levels" / "official"


# ---------------------------------------------------------------------------
# Level specs.  Each entry is (id, max_steps, ascii_grid, bfs_max_depth).
# Border walls are explicit so the entire grid is enclosed.
# ---------------------------------------------------------------------------

LEVELS: list[tuple[str, int, str, int]] = [
    # ============================================================ T1
    (
        "t1_01_baba_is_you", 25,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall baba .    .    .    .    flag wall
        wall .    .    .    .    .    .    wall
        wall .    .    FLAG IS   WIN  .    wall
        wall wall wall wall wall wall wall wall
        """,
        20,
    ),
    (
        "t1_02_step_up", 25,
        """
        wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    wall
        wall .    .    .    .    flag wall
        wall .    .    .    .    .    wall
        wall .    .    .    .    .    wall
        wall baba .    .    .    .    wall
        wall FLAG IS   WIN  .    .    wall
        wall wall wall wall wall wall wall
        """,
        25,
    ),
    (
        "t1_03_corner_eval", 25,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall baba .    .    .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall .    .    .    .    .    flag wall
        wall .    FLAG IS   WIN  .    .    wall
        wall wall wall wall wall wall wall wall
        """,
        25,
    ),
    # ============================================================ T2  (STOP walls)
    (
        "t2_01_wall_gap", 30,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall .    .    .    wall .    .    wall
        wall baba .    .    .    .    flag wall
        wall .    .    .    wall .    .    wall
        wall .    FLAG IS   WIN  .    .    wall
        wall WALL IS   STOP .    .    .    wall
        wall wall wall wall wall wall wall wall
        """,
        25,
    ),
    (
        "t2_02_around_wall", 30,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall WALL IS   STOP .    .    .    wall
        wall baba wall wall wall wall .    wall
        wall .    .    .    .    wall .    wall
        wall .    FLAG IS   WIN  wall flag wall
        wall .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall
        """,
        30,
    ),
    (
        "t2_03_corridor_eval", 30,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall WALL IS   STOP .    .    .    wall
        wall baba .    wall .    .    .    wall
        wall .    .    wall .    .    flag wall
        wall .    .    wall FLAG IS   WIN  wall
        wall .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall
        """,
        30,
    ),
    # ============================================================ T3  (PUSH rocks)
    (
        "t3_01_push_rock", 30,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall ROCK IS   PUSH .    .    .    wall
        wall baba rock .    .    .    flag wall
        wall .    .    .    .    .    .    wall
        wall FLAG IS   WIN  .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall
        """,
        25,
    ),
    (
        "t3_02_push_two", 30,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall ROCK IS   PUSH .    .    .    wall
        wall baba rock rock .    .    flag wall
        wall .    .    .    .    .    .    wall
        wall FLAG IS   WIN  .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall
        """,
        25,
    ),
    (
        "t3_03_push_eval", 30,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall ROCK IS   PUSH .    .    .    wall
        wall baba rock .    rock .    flag wall
        wall .    .    .    .    .    .    wall
        wall FLAG IS   WIN  .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall
        """,
        30,
    ),
    # ============================================================ T4  (break STOP rule)
    (
        "t4_01_break_stop", 35,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall baba .    wall .    .    flag wall
        wall .    .    wall .    .    .    wall
        wall .    .    wall .    .    .    wall
        wall WALL IS   STOP FLAG IS   WIN  wall
        wall wall wall wall wall wall wall wall
        """,
        30,
    ),
    (
        "t4_02_push_stop_aside", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall baba .    wall .    .    .    flag wall
        wall .    .    wall .    .    .    .    wall
        wall .    .    wall .    .    .    .    wall
        wall WALL IS   STOP .    .    FLAG IS   wall
        wall .    .    .    .    .    .    WIN  wall
        wall wall wall wall wall wall wall wall wall
        """,
        35,
    ),
    (
        "t4_03_break_stop_eval", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall baba .    .    wall .    .    flag wall
        wall .    .    .    wall .    .    .    wall
        wall .    .    .    wall .    .    .    wall
        wall WALL IS   STOP wall .    FLAG IS   wall
        wall .    .    .    .    .    .    WIN  wall
        wall wall wall wall wall wall wall wall wall
        """,
        40,
    ),
    # ============================================================ T5  (water sink)
    (
        "t5_01_water_detour", 35,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall WATER IS  SINK .    .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall baba .    water water .    .    flag wall
        wall .    .    .    .    .    .    .    wall
        wall .    FLAG IS   WIN  .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall wall
        """,
        30,
    ),
    (
        "t5_02_sink_with_rock", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall ROCK IS   PUSH .    .    .    .    wall
        wall WATER IS  SINK .    .    .    .    wall
        wall baba rock water .    .    .    flag wall
        wall .    .    .    .    .    .    .    wall
        wall .    FLAG IS   WIN  .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall wall
        """,
        30,
    ),
    (
        "t5_03_water_eval", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall WATER IS  SINK .    .    .    .    wall
        wall baba .    water .    .    .    flag wall
        wall .    .    water .    .    .    .    wall
        wall .    .    water .    .    .    .    wall
        wall .    FLAG IS   WIN  .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall wall
        """,
        35,
    ),
    # ============================================================ T6  (form WIN rule)
    (
        "t6_01_make_flag_win", 35,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall baba .    .    .    .    flag wall
        wall .    .    .    .    .    .    wall
        wall FLAG IS   .    WIN  .    .    wall
        wall .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall
        """,
        30,
    ),
    (
        "t6_02_make_rock_win", 40,
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    wall
        wall ROCK IS   PUSH .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall baba .    rock .    .    .    wall
        wall .    .    .    .    .    .    wall
        wall ROCK IS   .    WIN  .    .    wall
        wall wall wall wall wall wall wall wall
        """,
        30,
    ),
    (
        "t6_03_make_win_eval", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall baba .    .    .    .    .    flag wall
        wall .    .    .    .    .    .    .    wall
        wall FLAG .    IS   .    WIN  .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall wall wall wall wall wall wall wall wall
        """,
        35,
    ),
    # ============================================================ T7  (schema drift)
    (
        "t7_01_become_rock", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  ROCK IS   YOU  .    wall
        wall ROCK IS   PUSH .    .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall baba .    wall wall wall .    flag wall
        wall .    rock .    .    .    .    .    wall
        wall .    .    .    .    FLAG IS   WIN  wall
        wall wall wall wall wall wall wall wall wall
        """,
        35,
    ),
    (
        "t7_02_swap_you", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall baba .    wall .    .    .    flag wall
        wall .    .    wall .    .    .    .    wall
        wall .    .    wall .    .    rock .    wall
        wall ROCK IS   YOU  FLAG IS   WIN  .    wall
        wall wall wall wall wall wall wall wall wall
        """,
        35,
    ),
    (
        "t7_03_drift_eval", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall .    .    .    .    .    .    .    wall
        wall baba .    wall .    .    rock .    wall
        wall .    .    wall .    .    .    .    wall
        wall .    .    wall .    .    .    flag wall
        wall ROCK IS   YOU  .    FLAG IS   WIN  wall
        wall wall wall wall wall wall wall wall wall
        """,
        35,
    ),
    # ============================================================ T8  (OPEN / SHUT)
    (
        "t8_01_key_door", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall KEY  IS   OPEN .    .    .    .    wall
        wall DOOR IS   SHUT .    .    .    .    wall
        wall baba key  door .    .    .    flag wall
        wall .    .    .    .    .    .    .    wall
        wall .    FLAG IS   WIN  .    .    .    wall
        wall wall wall wall wall wall wall wall wall
        """,
        30,
    ),
    (
        "t8_02_grab_key_first", 40,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall KEY  IS   OPEN .    .    .    .    wall
        wall DOOR IS   SHUT .    .    .    .    wall
        wall baba .    door .    .    .    flag wall
        wall .    .    .    .    .    .    .    wall
        wall .    key  .    .    FLAG IS   WIN  wall
        wall wall wall wall wall wall wall wall wall
        """,
        35,
    ),
    (
        "t8_03_door_eval", 45,
        """
        wall wall wall wall wall wall wall wall wall
        wall BABA IS   YOU  .    .    .    .    wall
        wall KEY  IS   OPEN .    .    .    .    wall
        wall DOOR IS   SHUT .    .    .    .    wall
        wall baba .    door .    door .    flag wall
        wall .    .    .    .    .    .    .    wall
        wall .    key  .    key  FLAG IS   WIN  wall
        wall wall wall wall wall wall wall wall wall
        """,
        40,
    ),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    print(f"[build] writing {len(LEVELS)} levels to {OUT_DIR}")
    for lid, max_steps, grid_text, bfs_depth in LEVELS:
        rows = parse_grid(grid_text)
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            print(f"  [FAIL] {lid}: rows of inconsistent width {widths}")
            failures.append(lid)
            continue
        out = OUT_DIR / f"{lid}.txt"
        write_map(out, rows)
        # Verify solvable.
        world = parse_level({"map_path": str(out), "max_steps": max_steps})
        sol = bfs_solve(world, max_depth=bfs_depth, max_nodes=200_000)
        if sol is None:
            print(f"  [FAIL] {lid}: BFS could not solve within depth={bfs_depth}")
            failures.append(lid)
        else:
            print(f"  [ok]   {lid:30s}  bfs_len={len(sol):3d}  size={world.width}x{world.height}")
    if failures:
        print(f"\n{len(failures)} level(s) failed: {failures}")
        return 1
    print("\nAll levels written and BFS-verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
