# Architecture & Repository Structure

> Last updated: 2026‑04‑25. Round 2 of the Meta × OpenEnv Hackathon.

This document is the canonical map of the `baba-rlvr` codebase. It lists every
module, what it does, and how the pieces connect. Pair it with
[ROADMAP.md](ROADMAP.md) (gaps & next steps) and
[HANDOVER.md](HANDOVER.md) (continuation prompt).

---

## 1. Top-level layout

```
baba-rlvr/
├── pyproject.toml            # uv project, deps, console-scripts
├── uv.lock
├── README.md                 # pitch & quickstart (NOTE: still references pure-Python engine)
├── ARCHITECTURE.md           # ← this file
├── ROADMAP.md                # what's missing + plan
├── HANDOVER.md               # next-session prompt
│
├── src/baba_rlvr/            # main Python package
│   ├── engine/               # pyBaba adapter (World, Direction, ...)
│   ├── levels/               # .txt map loader + map-writer helper
│   ├── server/               # FastAPI OpenEnv contract + browser play
│   ├── reward/               # verifier-gated reward tracker
│   ├── memory/               # lessons.md scratchpad
│   ├── pcg/                  # MAP-Elites + BFS solver
│   ├── viz/                  # PIL renderer + GIF/strip CLI
│   ├── eval/                 # random/baseline agents
│   ├── training/             # GRPO trainer + prompts + reward callable
│   └── client/               # tiny HTTP client for the OpenEnv server
│
├── tests/                    # pytest suite (currently 19/19 green)
├── levels/templates/         # custom RLVR puzzles in baba-is-auto .txt format
├── vendor/baba-is-auto/      # cloned C++ engine (utilForever/baba-is-auto)
│   └── Extensions/BabaPython/  # pybind11 bindings — patched
├── scripts/build_pybaba.sh   # builds pyBaba via conda-forge toolchain
├── notebooks/                # Colab GRPO demo (needs refresh)
├── demo/                     # rendered PNG/GIF demo artifacts
├── memory_runs/              # runtime: lessons.md per episode (gitignored)
└── .venv/                    # uv-managed venv (gitignored)
```

---

## 2. Engine: `src/baba_rlvr/engine/`

### `engine/types.py`
Stable string-enum vocabulary used everywhere. Survives serialization to JSON
and into LLM prompts.
- `Direction` — `up/down/left/right/wait` plus `.delta` helper.
- `EntityKind` — `baba/rock/wall/flag/skull/lava/keke/door/key`.
- `WordKind` — every text token we surface (nouns, `IS`, properties).
- `Property` — `YOU/WIN/STOP/PUSH/KILL/DEFEAT/SINK/MELT/HOT`.
- `Tile` — frozen dataclass `(entities, words)` with `.render()` glyph.
- Tables: `NOUN_WORDS`, `PROPERTY_WORDS`, `VERB_WORDS`.

### `engine/world.py`  *(core adapter)*
Thin wrapper over `pyBaba.Game`. The C++ engine is the source of truth;
every public attribute on `World` is a derived Python projection refreshed
after each `step()`.

Key API:
- `World(map_path, max_steps)` — loads a baba-is-auto `.txt` map.
- `step(Direction)` → `info` dict (`died/moved/invalid_move/truncated`).
- `clone()` — replays the action history into a new `pyBaba.Game`.
- `parse_rules() / rules` — set of `(EntityKind, Property)` tuples.
- `you_entities() / win_entities() / stop_entities() / push_entities() / kill_entities()`.
- `grid: list[list[Tile]]`, `width`, `height`, `won`, `lost`, `step_count`.
- `render_ascii()`, `to_tokens()`.

Translation tables `_ENTITY_BY_ICON`, `_WORDKIND_BY_TEXT`,
`_PROPERTY_BY_TEXT` map the 176-member `pyBaba.ObjectType` flat enum to our
readable string enums; unknown / exotic ObjectTypes are silently dropped
from the projection (they still execute in C++).

### `engine/__init__.py`
Re-exports the public surface.

---

## 3. Vendored C++ engine: `vendor/baba-is-auto/`

Cloned with `--recursive` from `utilForever/baba-is-auto` (MIT). Submodules:
`pybind11`, `doctest`, `random`. Build is driven by its own `setup.py`
(CMake → produces `pyBaba.so`).

**Patches applied (do not lose on update!)**:
- `Extensions/BabaPython/Sources/Rules/RuleManager.cpp` — added
  `#include <pybind11/stl.h>` so `GetRules()` can convert
  `std::vector<Rule>` to a Python list.
- `Extensions/BabaPython/Sources/Rules/Rule.cpp` — added `stl.h` plus
  `def_readwrite("objects", &Rule::objects)` to expose the
  `(Object, Object, Object)` triple to Python.

**Build entry point**: `scripts/build_pybaba.sh` (uses a conda-forge
toolchain so no sudo is required).

**pyBaba surface (confirmed working)**:
- `Game(path)`, `.Reset()`, `.MovePlayer(Direction)`, `.GetMap()`,
  `.GetPlayState()`, `.GetPlayerIcon()`, `.GetRuleManager()`.
- `Map.GetWidth/GetHeight/At(x,y)/GetPositions(ObjectType)/AddObject/RemoveObject`.
- `Object.GetTypes/HasType/HasNounType/HasPropertyType/HasTextType/HasVerbType`.
- `RuleManager.GetRules(ObjectType)→list[Rule]`, `.FindPlayer()`,
  `.HasProperty(types, prop)`, `.AddRule/.RemoveRule/.ClearRules`,
  `.GetNumRules`.
- `Rule.objects` → `(Object, Object, Object)`.
- `Direction.{NONE,UP,DOWN,LEFT,RIGHT}`,
  `PlayState.{INVALID,PLAYING,WON,LOST}`.
- `Preprocess.StateToTensor(game)` — RL feature tensor.

---

## 4. Levels: `src/baba_rlvr/levels/`

### `levels/loader.py`
At import time scans two directories for `*.txt` maps (header `W H` then
`H` rows of space-separated `pyBaba.ObjectType` integer IDs):

- `levels/templates/` — handcrafted RLVR puzzles (4 levels):
  `tutorial_01`, `use_mention_01`, `schema_drift_01`, `self_redefine_01`.
- `vendor/baba-is-auto/Resources/Maps/` — upstream demo levels, registered
  with `vendor_` prefix (6 levels: `vendor_baba_is_you`, `vendor_off_limits`,
  `vendor_off_limits_bug`, `vendor_out_of_reach`, `vendor_simple_map`,
  `vendor_volcano`).

Public API:
- `LEVEL_REGISTRY: dict[str, dict]` — `{"map_path": "...", "max_steps": int}`.
- `load_level(level_id) -> World`.
- `register_level(level_id, spec)` — used by the PCG generator.

### `levels/map_writer.py`
Translates a 2D list of readable tokens (`"BABA"`, `"is"`, `"YOU"`,
`"baba"`, `"."`) into a baba-is-auto integer-ID `.txt` map. Used to
author handcrafted levels and (eventually) PCG output.

---

## 5. Server: `src/baba_rlvr/server/`

### `server/schemas.py`
Pydantic v2 models for the OpenEnv contract: `BabaAction`, `BabaObservation`
(`grid_ascii / grid_tokens / active_rules / you_entities / win_entities /
step_count / max_steps / level_id / memory_excerpt`), `Rule`, `StepResponse`,
`ResetRequest/ResetResponse`, `CloseResponse`, `ActionType` enum.

### `server/env.py`
`BabaEnv` — per-session state holding `World` + `RewardTracker` + optional
`MemoryStore`. Methods: `reset()`, `step(BabaAction)`, `episode_summary()`.
Internal `_observe()` builds a `BabaObservation` from the current `World`.

### `server/main.py`
FastAPI app:
- `POST /reset` (level_id, use_memory, seed) → session
- `POST /step/{sid}`, `GET /state/{sid}`, `POST /close/{sid}`
- `GET /levels`, `GET /health`
- `GET /play` — single-page browser game (HTML below)
- `GET /play/frame/{sid}.png` — PNG of current state via `viz.renderer`
- `GET /play/solve/{level_id}` — BFS solution for the given level

`run()` is the `baba-server` console script entry point.

### `server/static/play.html`
Vanilla-JS SPA with: level dropdown, arrow-key controls, BFS-solve animation
button, sidebar showing live rules / YOU / WIN / cumulative return.
Persists `session_id` in `sessionStorage` and re-fetches the frame PNG after
every step.

---

## 6. Reward & memory

### `reward/tracker.py`
The **verifier**. Reads only structured engine state (rules, you-entities,
won/died/invalid flags) — never LLM text — so the agent cannot fabricate
signal through prompt manipulation.

Constants: `WEIGHT_WIN=10, WEIGHT_DEATH=-2, WEIGHT_INVALID=-0.5, WEIGHT_STEP=-0.01`.

Milestones (each fires **at most once** per episode): `first_rule_break`,
`first_rule_make`, `self_redefine`, `win_condition_made`, `neutralized_kill`.
Anti-hacking: `triggered: set[str]` blocks re-fires; `seen_rule_signatures:
set[(str,str)]` blocks toggle-loop farming.

### `memory/store.py`
Hybrid agentic memory: weights update via GRPO **and** a verifier-gated
markdown scratchpad (`memory_runs/<level>/lessons.md`). The LLM can *read*
the file via `BabaObservation.memory_excerpt` but only the verifier appends
entries through `note_milestone(name, obs)` and `flush(won=)`. Capped at
`MAX_MEMORY_LINES=20`. Toggle via `reset(use_memory=True)`.

---

## 7. Procedural generation: `src/baba_rlvr/pcg/`

### `pcg/solver.py`
BFS over `World` states. `state_key(world)` = tuple-of-tuples of
`(tile.entities, tile.words)`; `step_count` excluded so the same
configuration reached by different paths collapses. `bfs_solve(world,
max_depth=30, max_nodes=50_000)` returns a shortest action sequence or
`None`.

> ⚠️ Each `world.clone()` replays the full action history (no
> `Game.Clone()` exists upstream). For depth-15 BFS with the small custom
> levels this is fine (<1 s); deeper search may need the
> `Preprocess.StateToTensor(game)` snapshot for hashing instead.

### `pcg/map_elites.py`  ⚠️ **BROKEN — needs rewrite**
Uses the **old YAML-rows spec format** (`spec["rows"]`, `"."`, `"#"`,
`"BABA"`). After the engine pivot, specs are `{"map_path", "max_steps"}`.
The mutator must operate on tokenized 2D grids and write `.txt` maps via
`levels.map_writer`. See [ROADMAP.md](ROADMAP.md) §1.

### `pcg/cli.py`
Typer CLI: `uv run baba-pcg generate ...`, `... show archive.pkl`. Currently
broken transitively because of `map_elites.py`.

---

## 8. Visualization: `src/baba_rlvr/viz/`

### `viz/renderer.py`
PIL-based renderer. No sprite assets — draws shapes + colored letters
with DejaVu/Arial. Compatible with the new `World`:
- `render_world(world, *, cell, sidebar_w, title, action_taken, reward, step_idx)` → `PIL.Image`
- `render_trajectory_gif(frames, out, ...)` → animated GIF
- `render_trajectory_strip(frames, out, ...)` → multi-panel PNG
- `rollout_actions(world, actions)` → list of `(world_clone, action, info)`
  frames (terminates on `won`/`lost`)
- `actions_from_strings(["u","r",...])` → `list[Direction]`

> 💡 The vendored sprites live at
> `vendor/baba-is-auto/Extensions/BabaRL/baba-babaisyou-v0/sprites/` with a
> `pygame`-based reference renderer. Wiring them in (optional) is in
> [ROADMAP.md](ROADMAP.md) §4.

### `viz/cli.py`
Typer CLI exposed as `baba-viz`: `frame`, `play`, `solve`, `strip`
sub-commands.

---

## 9. Training: `src/baba_rlvr/training/`

### `training/prompts.py`
- `SYSTEM_PROMPT` — describes the OpenEnv contract + JSON action format.
- `build_prompt(obs)` — turns a `BabaObservation` into the user message
  (rules, YOU/WIN, ASCII grid, optional memory excerpt).
- `parse_action(text)` — JSON-first regex fallback to extract the next
  action token.

### `training/reward_callable.py`
TRL `GRPOTrainer`-compatible reward function. Each prompt represents the
*initial* state of an episode; the completion contributes the first action;
the rest of the episode is driven via an injected `GenerateFn` against a
running env server. Returns the trajectory return as the scalar reward.

### `training/grpo_train.py`
Single-GPU Colab-friendly training entry point (`uv run python -m
baba_rlvr.training.grpo_train --env-url ... --model unsloth/Qwen2.5-3B-Instruct-bnb-4bit`).
Optional `--smoke` for CPU pipeline tests. Loads either the built-in level
templates or a MAP-Elites archive (`--curriculum levels/archive.pkl`).
**Not yet end-to-end tested with the new engine.**

---

## 10. Eval & client

### `eval/random_agent.py`, `eval/cli.py`
`baba-eval` CLI — runs N random rollouts per level, reports success rate
and average return. Useful baseline for the deck.

### `client/baba_client.py`
Tiny `httpx`-based wrapper around the OpenEnv contract. Used by the
training loop and could be re-used by external evaluation scripts.

---

## 11. Tests: `tests/`

19 tests, all passing post-pivot.

- `test_engine.py` — level loads, hand-computed solution wins, rule parsing.
- `test_reward.py` — step cost, milestone fires once, win dominates,
  loop-farming blocked.
- `test_solver.py` — BFS finds the tutorial solution.
- `test_server.py` — health, levels listed, full episode through HTTP, 404
  on unknown session, `/play` page served, `/play/frame/{sid}.png` PNG
  bytes, `/play/solve/{lvl}` returns actions.
- `test_viz.py` — `render_world`, GIF round-trip, strip layout, rollout
  stops at terminal.

---

## 12. Build & runtime contract

| Concern              | Status |
| -------------------- | ------ |
| Python venv          | `uv sync --extra dev` |
| C++ pyBaba           | `bash scripts/build_pybaba.sh` (conda-forge toolchain assumed at `~/miniconda3/envs/babaenv/`) |
| Run server           | `uv run baba-server` (defaults `0.0.0.0:8000`) |
| Browser play         | open `http://localhost:8000/play` |
| Tests                | `uv run pytest -q` |
| Generate curriculum  | `uv run baba-pcg generate` *(currently broken — see ROADMAP)* |
| Train (GPU)          | `uv run python -m baba_rlvr.training.grpo_train ...` *(needs end-to-end run)* |

Environment variables consumed: `BABA_HOST`, `BABA_PORT`, `BABA_MEMORY_DIR`.
