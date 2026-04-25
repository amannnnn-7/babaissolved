# Baba Is You — RLVR Environment

> Submission for the **Meta x OpenEnv Hackathon** (Scaler School of Technology, April 25–26 2026).
> Round 2 — Round 2 graded on Environment Innovation (40%), Storytelling (30%), Reward
> Improvement (20%), and Training-script coherence (10%).

## Pitch in one paragraph

*Baba Is You* is a puzzle game where the **rules of physics are themselves movable
objects on the grid** — pushing the words `BABA`, `IS`, `YOU` together makes "Baba"
the player; pushing `WALL`, `IS`, `STOP` makes walls solid. We turn this into an
**RLVR (Reinforcement Learning with Verifiable Rewards)** training environment for
LLM agents: the agent must reason about, and *rewrite*, the rules of the world it
inhabits. This directly probes:

- **Long-Horizon Planning** — solutions are 20–60 deep moves with delayed reward.
- **Schema Drift** (Patronus AI sub-theme) — the agent's own identity (`X IS YOU`)
  can change mid-episode, mirroring real-world API/policy contract changes.
- **Self-Improvement** — MAP-Elites procedural generation produces an infinite,
  difficulty-stratified curriculum, plus an optional **agentic memory scratchpad**
  (`lessons.md`) the agent reads/writes across episodes.

Training uses **GRPO** (Group Relative Policy Optimization) via Hugging Face TRL +
Unsloth, talking to the env over Meta's **OpenEnv** HTTP contract.

## Architecture

```
┌──────────────────────────┐  HTTP   ┌────────────────────────────────────┐
│  GRPOTrainer (TRL)       │────────▶│  OpenEnv FastAPI server            │
│   • Unsloth Qwen2.5-3B   │         │   • baba_rlvr.engine (pure-Python  │
│   • LoRA, 4-bit          │         │     deterministic Baba ruleset)    │
│   • K=8 group rollouts   │◀────────│   • RewardTracker (verifier)       │
└──────────────────────────┘ rewards └────────────────────────────────────┘
            ▲                                           ▲
            │ curriculum                                │ levels
            │                                           │
        ┌───┴───────────────────────────────────────────┴────┐
        │  MAP-Elites PCG  +  BFS solver  +  MemoryStore     │
        └────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# 1. Install (CPU is fine for env server + tests)
uv sync

# 2. Run the env server
uv run baba-server                     # http://localhost:8000/docs

# 3. Sanity check with a random agent
uv run python -m baba_rlvr.eval.random_agent --episodes 20

# 4. Generate a curriculum
uv run baba-pcg generate --iterations 5000 --out levels/archive.pkl

# 5. Visualize a level / trajectory / BFS-solution
uv run baba-viz frame tutorial_01 --out demo/tut.png
uv run baba-viz solve schema_drift_01 --out demo/sd.gif
uv run baba-viz strip tutorial_01 --actions rrrrrr --out demo/strip.png

# 6. Train (Colab / GPU box)
uv sync --extra train --extra train-unsloth
uv run python -m baba_rlvr.training.grpo_train
```

## Repository layout

```
src/baba_rlvr/
├── engine/        # Pure-Python Baba Is You ruleset (no C++ dep, deterministic)
├── server/        # FastAPI + Pydantic OpenEnv contract
├── client/        # HTTPEnvClient subclass for trainers
├── pcg/           # MAP-Elites procedural generation + BFS solver
├── memory/        # Agentic scratchpad (lessons.md) gated by verifier
├── reward/        # RewardTracker — milestone verifier
├── training/      # GRPO trainer (TRL + Unsloth)
├── viz/           # Pillow-based trajectory & gameplay visualizer
└── eval/          # Random / heuristic / trained-agent eval harnesses
levels/            # Hand-built test levels + generated archives
notebooks/         # Colab demo + ablation plots
tests/             # pytest suite
```

## Judging-criteria mapping

| Criterion (weight) | Where to look |
|---|---|
| Environment Innovation (40%) | `engine/` (rules-as-objects), `levels/templates/` (use-mention, schema-drift, self-sacrifice) |
| Storytelling (30%) | `notebooks/demo.ipynb`, blog post, 3-min video |
| Reward Improvement (20%) | `notebooks/ablation_memory.ipynb`, W&B reward curves, with-vs-without-memory table |
| Reward & Training Logic (10%) | `reward/tracker.py`, `training/grpo_train.py` |

## License

MIT.
