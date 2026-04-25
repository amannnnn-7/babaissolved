# Roadmap — What's Missing & How to Finish It

> Companion to [ARCHITECTURE.md](ARCHITECTURE.md). Snapshot at 2026‑04‑25, after
> the pivot to the C++ `pyBaba` engine. All items below were either invalidated by
> that pivot or were never finished in the first scaffold.

Priority key: **P0** = blocks the Round 2 submission, **P1** = needed for a
strong demo, **P2** = nice-to-have / post-hackathon.

---

## P0 — Blockers for the Round 2 deliverable

### 1. Rewrite `pcg/map_elites.py` for the new spec format
**Current state.** Mutator operates on `spec["rows"]` (the YAML token-string
format from before the pivot). After the engine pivot, registry specs are
`{"map_path", "max_steps"}` only — there is no `rows` field. Importing
`map_elites` works, but `run_map_elites()` will `KeyError` on the first
mutation, breaking `baba-pcg generate`.

**What to do.**
1. Add a helper `levels.map_writer.read_map(path) -> list[list[str]]` that
   reads a `.txt` map and returns the readable token grid (inverse of
   `write_map`).
2. Rewrite `_mutate_spec` to operate on the token grid (`list[list[str]]`):
   - `swap_cells`, `place_wall` (`"wall"`), `remove_wall`, `shift_rule`
     (find an `"IS"` token; swap its neighbours).
3. After mutation, materialize the child grid to a temp `.txt` under
   `levels/_generated/pcg_<id>.txt`, then call `parse_level({"map_path":
   tmp, "max_steps": ...})` for solver verification.
4. Persist `Elite.spec` as `{"map_path": absolute_str, "max_steps": int}`
   so it round-trips through `register_level`.
5. Make sure `_count_rules_modified` still works against the new `World`
   (it uses `world.rules` which is preserved — should require zero changes).

**Acceptance.** `uv run baba-pcg generate --iterations 200` produces a
non-empty archive with multiple difficulty cells; `baba-pcg show` prints
it; `tests/test_pcg.py` (new) covers ≥1 mutator + a 50-iteration archive
smoke test.

### 2. Refresh `README.md`
The README still says “pure-Python engine, ~30 µs/step”. Must call out:
- Dependency on the vendored C++ `pyBaba` (and submodules).
- Build prerequisite (`scripts/build_pybaba.sh`, conda-forge toolchain note).
- Updated quickstart (`uv sync && bash scripts/build_pybaba.sh && uv run
  baba-server`).
- Pointer to `ARCHITECTURE.md` / `ROADMAP.md`.

### 3. Regenerate demo artifacts
`demo/tutorial_initial.png`, `demo/tutorial_solution.gif`,
`demo/tutorial_strip.png` were generated against the old engine. Re-run:

```bash
uv run baba-viz frame  --level tutorial_01 --out demo/tutorial_initial.png
uv run baba-viz solve  --level tutorial_01 --out demo/tutorial_solution.gif
uv run baba-viz strip  --level tutorial_01 --out demo/tutorial_strip.png
```

Spot-check the rendered rules sidebar matches the `pyBaba`-derived rules.

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

### 6. Sprite-based rendering (optional but eye-catching)
Vendored sprites at `vendor/baba-is-auto/Extensions/BabaRL/baba-babaisyou-v0/sprites/`.
Add `viz.sprites_renderer.SpriteRenderer` that:
- Loads PNG sprites by `pyBaba.ObjectType` name.
- Falls back to the shape renderer when a sprite is missing.
- Is opt-in via `render_world(..., backend="sprites" | "shapes")`.

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

### 9. Docstring sweep + ruff/mypy
The new `engine/world.py`, `levels/loader.py`, `levels/map_writer.py`
files are documented; legacy modules still reference the pure-Python
engine. Run `uv run ruff check . && uv run mypy src` and clean the
fall-out (mostly unused imports + outdated docstrings in `engine/types.py`).

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
| `engine.World` (adapter)        | ✅    | ✅     | 19/19 tests green |
| Custom `.txt` levels (4)        | ✅    | ✅     | tutorial / use-mention / schema-drift / self-redefine |
| Vendor levels (6)               | ✅    | smoke  | load + run; not in test suite |
| FastAPI OpenEnv server          | ✅    | ✅     | reset/step/state/close + /play/* |
| Browser play (`play.html`)      | ✅    | smoke  | arrow keys + BFS-solve animation |
| RewardTracker                   | ✅    | ✅     | unchanged after pivot |
| MemoryStore (lessons.md)        | ✅    | manual | works; no automated test |
| BFS solver                      | ✅    | ✅     | tutorial 6 steps; schema-drift 11 steps |
| Visualizer (`viz/renderer`)     | ✅    | ✅     | shapes-only |
| `viz/cli` (`baba-viz`)          | ✅    | manual |       |
| `eval/cli` (`baba-eval`)        | ✅    | manual |       |
| `MAP-Elites` PCG                | ❌    | ❌     | **broken — see §1** |
| GRPO training (`grpo_train.py`) | ⚠️    | ❌     | runs but never end-to-end on new engine |
| Colab notebook                  | ⚠️    | ❌     | needs refresh |
| Demo artifacts (`demo/`)        | ⚠️    | ❌     | stale, regenerate |
| README                          | ⚠️    | n/a    | references pure-Python engine |
