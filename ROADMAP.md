# Roadmap — What's Missing & How to Finish It

> Companion to [ARCHITECTURE.md](ARCHITECTURE.md). Snapshot at 2026‑04‑25, after
> the pivot to the C++ `pyBaba` engine. All items below were either invalidated by
> that pivot or were never finished in the first scaffold.

Priority key: **P0** = blocks the Round 2 submission, **P1** = needed for a
strong demo, **P2** = nice-to-have / post-hackathon.

---

## P0 — Blockers for the Round 2 deliverable

### 1. Rewrite `pcg/map_elites.py` for the new spec format — ✅ done
`levels.map_writer.read_map()` now inverts `.txt` maps into readable token
grids, `_mutate_spec()` operates on `{"map_path", "max_steps"}` specs, generated
children are written under `levels/_generated/`, and elites persist absolute
`map_path` specs. `tests/test_map_writer.py` and `tests/test_pcg.py` cover the
round-trip and PCG smoke path.

### 2. Refresh `README.md` — ✅ done
The README now calls out the vendored C++ `pyBaba` engine, uv setup, optional
`scripts/build_pybaba.sh`, levels, visualizer, PCG, and GRPO smoke/full commands.

### 3. Regenerate demo artifacts — ✅ done
`demo/tutorial_initial.png`, `demo/tutorial_solution.gif`,
`demo/tutorial_strip.png` were generated against the old engine. Re-run:

```bash
uv run baba-viz frame tutorial_01 --out demo/tutorial_initial.png
uv run baba-viz solve tutorial_01 --out demo/tutorial_solution.gif
uv run baba-viz strip tutorial_01 --out demo/tutorial_strip.png
```

Generated locally under `demo/` from the current pyBaba-backed engine.

### 4. End-to-end GRPO smoke test
`training/grpo_train.py --smoke` was scaffolded but never run against the
new engine. Steps:
1. Start the server: `uv run baba-server &`
2. Run smoke: `uv run python -m baba_rlvr.training.grpo_train --smoke
   --env-url http://localhost:8000`
3. Confirm: dataset rows load, `make_reward_func` produces non-`NaN`
   rewards, no schema mismatches between client and server.

If/when this works on CPU, do the same in `notebooks/colab_grpo_demo.ipynb`
(currently 111 lines; verify it imports the package without referencing
the old YAML format).

---

## P1 — Needed for a strong demo

### 5. Update `notebooks/colab_grpo_demo.ipynb`
Likely references YAML-format levels and the legacy renderer signature.
Walk through and update to:
- Install via `uv pip install -e .` and (optionally) `bash
  scripts/build_pybaba.sh` (or fall back to `apt install build-essential
  cmake` for hosted Colab where conda isn't available).
- Render before/after rollouts using the new `viz` API.
- Train 50–200 GRPO steps on a single A100/L4 and plot return.

### 6. Sprite-based rendering — ✅ done
`viz.renderer` now uses vendored `baba-is-auto` sprites from
`Extensions/BabaGUI/sprites/` by default, falls back to the shape renderer when
a sprite is missing, and keeps `backend="sprites" | "shapes"` for CLI/debug use.

### 7. Build robustness
The current build assumes a conda env at `~/miniconda3/envs/babaenv/`.
Make the build path-tolerant:
- `scripts/build_pybaba.sh` already errors out cleanly if `cmake` isn't
  found; add an `apt`-based fallback for systems with sudo.
- Add a CI workflow (`.github/workflows/ci.yml`) that installs
  `build-essential cmake`, builds `pyBaba`, and runs `pytest`.
- Document that Colab's image already has `g++` & `cmake` so the conda
  step isn't required there.

### 8. Tests for the new surface
Add coverage for:
- `tests/test_levels.py` — `LEVEL_REGISTRY` contains both custom and
  `vendor_*` levels; every map loads to a `World` with consistent
  `width × height`.
- `tests/test_world_clone.py` — replaying a 5-action sequence from a
  cloned World produces the same `state_key` as the original.
- `tests/test_pcg.py` — mutator round-trips a `.txt` map; `run_map_elites
  --iterations 50` populates ≥3 cells.
- `tests/test_map_writer.py` — `tokenize/write_map/read_map` round-trip.
- `tests/test_mechanics.py` — early-pack mechanics smoke coverage for visible
  water, `AND` expansion, `SINK`, and `OPEN`/`SHUT`.

### 9. Docstring sweep + ruff/mypy
The pyBaba adapter, loader, map writer, renderer, and training entrypoint are
documented. Full-suite ruff still includes unrelated enum-style modernization
noise; use focused ruff selections while stabilizing the hackathon path.

---

## P2 — Polish / post-hackathon

### 10. Faster `World.clone()`
History-replay is O(N) per clone, making BFS O(depth²). Two options:
1. Add a `Game::Clone()` to the upstream C++ library (small patch — uses
   the implicit copy-ctor on `Game{m_map, m_ruleManager, m_playState}`)
   and bind it.
2. Use `pyBaba.Preprocess.StateToTensor(game)` as a hash key and skip the
   actual clone when only hashing is needed.

### 11. NOUN-IS-NOUN transformations in the verifier
The C++ engine resolves `WALL IS WATER` automatically, but our verifier
projection only sees NOUN-IS-PROPERTY rules. Add transformation events to
the milestone vocabulary (`first_transformation`, `transformed_player`).

### 12. OpenEnv compliance audit
Cross-check `server/main.py` against Meta's published OpenEnv spec — make
sure response shapes / field names exactly match (e.g. `terminated` vs
`done`, `truncated`). Important for being scored on Reward-Improvement.

### 13. Multi-agent / agent-vs-agent scenarios
Baba supports multiple YOU entities. Could expose a multi-agent variant
(`MultiBabaEnv`) where two LLMs each control a different YOU subset.

### 14. Save/load full episode trajectories
For Round 2 storytelling we want to record an "epic episode" and replay
it server-side. Add `POST /record/{sid}` and `GET /trajectory/{sid}.json`
endpoints, plus a `viz` command that builds a strip from a trajectory file.

---

## Verification matrix (what currently works)

| Component                       | Works | Tested | Notes |
| ------------------------------- | :---: | :----: | ----- |
| `pyBaba` build (patched)        | ✅    | ✅     | conda toolchain only path so far |
| `engine.World` (adapter)        | ✅    | ✅     | 29/29 tests green |
| Custom `.txt` levels (4)        | ✅    | ✅     | tutorial / use-mention / schema-drift / self-redefine |
| Vendor levels (6)               | ✅    | smoke  | load + run; not in test suite |
| FastAPI OpenEnv server          | ✅    | ✅     | reset/step/state/close + /play/* |
| Browser play (`play.html`)      | ✅    | smoke  | arrow keys + BFS-solve animation |
| RewardTracker                   | ✅    | ✅     | unchanged after pivot |
| MemoryStore (lessons.md)        | ✅    | manual | works; no automated test |
| BFS solver                      | ✅    | ✅     | tutorial 6 steps; schema-drift 11 steps |
| Visualizer (`viz/renderer`)     | ✅    | ✅     | sprites with shape fallback |
| `viz/cli` (`baba-viz`)          | ✅    | manual |       |
| `eval/cli` (`baba-eval`)        | ✅    | manual |       |
| `MAP-Elites` PCG                | ✅    | ✅     | `.txt` map mutation + BFS verification |
| GRPO training (`grpo_train.py`) | ✅    | smoke  | generated-curriculum smoke passes; A100 script ready |
| Colab notebook                  | ⚠️    | ❌     | needs refresh |
| Demo artifacts (`demo/`)        | ✅    | smoke  | regenerated locally |
| README                          | ✅    | n/a    | updated for pyBaba setup |
