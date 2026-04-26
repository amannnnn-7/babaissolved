# Handover Prompt

> Copy the entire fenced block below into the next Copilot / Claude session
> as the very first user message. It bootstraps the new agent with full
> context of where we are and what to do next.

---

```
You are continuing work on the **Baba Is You — RLVR** environment, my
submission to the Meta × OpenEnv Hackathon (Scaler School of Technology,
April 25–26 2026), Round 2.

## Project location
- Working directory: ~/babaissolved
- Python env: uv-managed venv at .venv/  (Python 3.11)
- `uv` may be installed at `$HOME/.local/bin/uv`; export that PATH first.
- C++ engine: vendored at vendor/baba-is-auto/ (utilForever/baba-is-auto, MIT)
- Build scripts: scripts/setup_vendor.sh and scripts/build_pybaba.sh.

## What this project is
A FastAPI **OpenEnv** server that exposes *Baba Is You* (a puzzle game where
the rules of physics are themselves movable objects on the grid) as an
RLVR training environment for LLM agents. Reward is verifier-gated through
hand-defined milestones (first_rule_break, self_redefine, win_condition_made,
etc.). Trained with **GRPO** via TRL + Unsloth (default run script uses
Qwen/Qwen3-4B-Instruct-2507; override `MODEL_ID` for another HF Qwen checkpoint).
**MAP-Elites PCG** generates a difficulty-stratified curriculum. Optional
**agentic memory scratchpad** (verifier-gated lessons.md) is a hybrid with
weight updates.

## Recent history (most important context)
1. We initially shipped a pure-Python Baba engine. It was bug-prone (push
   chains, NOUN-IS-NOUN transforms incomplete), so we **pivoted** to wrap
   the battle-tested C++ `pyBaba` library from utilForever/baba-is-auto.
2. We patch upstream pybind11 bindings and small C++ gameplay logic through
   `scripts/patches/baba-is-auto-bindings.patch` — see ARCHITECTURE.md §3.
3. We rewrote `src/baba_rlvr/engine/world.py` as a thin adapter over
   `pyBaba.Game` while preserving the same public `World` API (so the
   server, reward tracker, renderer, solver, and tests didn't need
   sweeping changes). The current suite has 29 tests.
4. Levels were converted from YAML token-rows to baba-is-auto **.txt
   integer-ID maps** under `levels/templates/*.txt`. The loader now also
   exposes vendored upstream maps with a `vendor_` prefix and generated maps
   under `levels/_generated`.
5. Early-pack mechanics now include visible water and `AND`, `SINK`, and
   `OPEN`/`SHUT` behavior tested in `tests/test_mechanics.py`.

## Required reading before you do ANYTHING
Open these three files at the repo root **in this order** and read them
fully:
  1. ARCHITECTURE.md  — module-by-module map of the codebase.
  2. ROADMAP.md       — exactly what is missing, prioritized P0/P1/P2.
  3. README.md        — current pitch and setup commands.

Then run `uv run pytest -q` to confirm the baseline (should be 29/29 green).
If pytest fails because pyBaba isn't importable, run
`bash scripts/build_pybaba.sh` first.

## Current next task
Use ROADMAP's verification matrix to track any remaining polish. The P0
setup/PCG/docs/demo path is complete; the current priority is presentation
and any extra mechanics tests needed for the chosen demo levels.

## Engineering ground rules
- Keep vendor changes reproducible through
  `scripts/patches/baba-is-auto-bindings.patch`.
- Keep the public `World` API stable. Downstream consumers (server/env.py,
  reward/tracker.py, viz/renderer.py, pcg/solver.py) rely on:
  `step / clone / parse_rules / rules / you_entities / win_entities /
   stop_entities / push_entities / kill_entities / grid / width / height /
   won / lost / step_count / max_steps / render_ascii / to_tokens`.
- Always run `uv run pytest -q` after a change. New work needs new tests.
- The OpenEnv contract lives in `server/schemas.py` — do not add fields
  without a deliberate reason; they ship to LLM prompts.
- Reward shaping is in `reward/tracker.py` and is the verifier of the RLVR
  story — keep it provable: read engine state, never LLM text.

## Useful commands
```
# Build C++ engine after first clone or upstream update
bash scripts/build_pybaba.sh

# Tests
uv run pytest -q

# Run the server (reloads on file change with --reload via uvicorn)
uv run baba-server

# Open browser-play
xdg-open http://localhost:8000/play

# CLI smoke
uv run baba-viz frame tutorial_01 --out /tmp/t.png
uv run baba-eval random --level-id tutorial_01 --episodes 20
uv run baba-pcg generate --iterations 200

# GRPO smoke (CPU)
uv run python -m baba_rlvr.training.grpo_train --smoke --env-url http://localhost:8000
```

## When in doubt
- Look first in ARCHITECTURE.md §N (it cross-links every module).
- Check tests/ for the expected contract of each module.
- The patch we applied to vendor/baba-is-auto bindings can be re-derived
  by reading session notes in this prompt; do not lose it on
  `git submodule update`.

Begin by opening ARCHITECTURE.md and ROADMAP.md, then propose a 5-step
plan for the next hour.
```
