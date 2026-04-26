---
title: Baba Is You RLVR
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 8000
base_path: /play
pinned: false
license: mit
tags:
  - openenv
  - fastapi
  - reinforcement-learning
  - rlvr
---

# Baba Is You for RLVR

An OpenEnv environment where the agent has to solve puzzles by rewriting the
rules of the world it lives in. In Baba Is You, text is physics: pushing
`BABA IS YOU`, `FLAG IS WIN`, or `WALL IS STOP` literally changes what the
grid means. That makes the environment a good fit for training language models
on long-horizon planning, rule manipulation, schema drift, and verifiable
reasoning under delayed reward.

This repository now includes the files needed for a Hugging Face Docker Space
and OpenEnv deployment: `openenv.yaml`, `Dockerfile`, and a top-level
`server.py` ASGI entrypoint that exposes the existing FastAPI app.

## Submission Links

Fill the live URLs below before final submission.

| Asset | Link |
|---|---|
| Hugging Face Space | TODO before submission: add the live `https://huggingface.co/spaces/...` URL |
| Colab / notebook | [notebooks/colab_grpo_demo.ipynb](notebooks/colab_grpo_demo.ipynb) |
| Main training launcher | [scripts/run_qwen_a100_grpo.sh](scripts/run_qwen_a100_grpo.sh) |
| Blog draft | [blog_draft.md](blog_draft.md) |
| Submission checklist | [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) |
| Architecture notes | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Training evidence plots | TODO after current run finishes: add committed PNGs or public W&B run links |
| Mini-blog / video | TODO before submission: add Hugging Face post or public YouTube link |

## Why This Environment Is Interesting

Most small grid environments test navigation. Baba Is You tests something more
interesting: whether an agent can reason about a mutable world model. The same
object can become controllable, deadly, or irrelevant depending on which word
tiles the agent pushes together.

That directly probes the hackathon criteria:

- Environment innovation: the agent is not only solving a puzzle, it is
  rewriting the rules that define the puzzle.
- Storytelling: the game is visually legible to a non-technical audience;
  judges can watch rules change in the browser and understand why behavior
  improves.
- Reward improvement: the reward is verifier-based and tied to concrete world
  events, so training curves are interpretable.
- Training pipeline: the environment speaks the OpenEnv HTTP contract and is
  trained end-to-end with GRPO through TRL + Unsloth.

## Environment Design

The FastAPI server exposes a standard OpenEnv-style session API:

- `POST /reset`
- `POST /step/{session_id}`
- `GET /state/{session_id}`
- `POST /close/{session_id}`
- `GET /levels`
- `GET /health`

For demos and judging, the same server also exposes a browser UI:

- `GET /play` for manual play
- `GET /play/frame/{session_id}.png` for rendered frames
- `GET /play/solve/{level_id}` for a BFS solution preview

Each observation includes the ASCII grid, tokenized grid, active rules,
current `YOU` entities, current `WIN` entities, step counters, level id, and
an optional verifier-gated memory excerpt.

## Reward Design

The reward function is intentionally verifiable. It only reads structured game
state from the engine and never trusts model text.

Base shaping:

- win: `+10.0`
- death: `-2.0`
- invalid action: `-0.5`
- step cost: `-0.01`

One-shot milestone rewards:

- `first_rule_break`: `+1.0`
- `first_rule_make`: `+1.0`
- `self_redefine`: `+2.0`
- `win_condition_made`: `+1.5`
- `neutralized_kill`: `+1.0`

Anti-hacking constraints:

- every milestone can fire at most once per episode,
- re-making an already seen rule gives no extra reward,
- loops are punished by the step cost,
- the verifier consumes engine state, not free-form model output.

## Levels and Curriculum

The repository has three complementary level sources:

1. Hand-authored RLVR templates in `levels/templates/` for core mechanics and
   quick demos.
2. An official 24-level curriculum in `levels/official/`, split into 8 tiers
   with 2 train levels and 1 held-out eval level per tier.
3. Solver-checked procedural levels produced by MAP-Elites under
   `levels/_generated/` when generation is run locally.

The 8 official tiers are:

1. Movement
2. STOP walls
3. PUSH rocks
4. Break a STOP rule
5. Water / SINK
6. Form a WIN rule
7. Schema drift (`IS YOU` changes)
8. OPEN / SHUT (key / door)

The curriculum trainer advances tier by tier when the rolling win rate on the
current tier reaches `0.6` over at least `32` episodes.

## Training Pipeline

There are two training entry points in the repo.

Minimal Colab-friendly GRPO demo:

- script: `python -m baba_rlvr.training.grpo_train`
- default model: `Qwen/Qwen3-4B-Instruct-2507`
- optimizer stack: TRL GRPO + Unsloth + LoRA + 4-bit loading
- defaults: `steps=200`, `num_generations=8`, `batch_size=4`,
  `grad_accum=2`, `learning_rate=5e-6`, `lora_r=32`, `lora_alpha=64`

Main hackathon A100 curriculum run:

- launcher: [scripts/run_qwen_a100_grpo.sh](scripts/run_qwen_a100_grpo.sh)
- trainer: `python -m baba_rlvr.training.curriculum_train`
- default model: `Qwen/Qwen2.5-1.5B-Instruct`
- defaults: `steps=150`, `num_generations=8`, `batch_size=4`,
  `grad_accum=2`, `learning_rate=5e-6`, `max_turns=30`,
  `max_prompt_length=2048`, `max_completion_length=2048`
- LoRA setup: `r=16`, `alpha=32`
- model loading: 4-bit + fp16 to avoid the bf16 mismatch hit in an earlier
  Qwen3 4B run

The launcher script also:

1. builds or verifies the official level pack,
2. starts the OpenEnv server,
3. trains the adapter,
4. runs base-vs-trained evaluation on train and held-out levels.

## Training Evidence

The current A100 training run is still in progress, so this README is wired
for submission but does not yet claim final metrics.

Artifacts produced by the active pipeline:

- local trajectories: `runs/qwen-curriculum-trajectories/trajectories.jsonl`
- checkpoint output: `ckpt-baba-curriculum/`
- eval outputs: `runs/eval/`
- W&B project: `baba-rlvr`

Expected public evidence to add before submission:

1. reward curve PNG with labeled axes,
2. loss curve PNG with labeled axes,
3. held-out base-vs-trained comparison,
4. public W&B run link or committed static plots,
5. mini-blog or short video URL.

## Run Locally

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --extra dev
bash scripts/build_pybaba.sh

uv run baba-server
```

Then open `http://localhost:8000/play`.

Useful local commands:

```bash
uv run baba-eval random --level-id tutorial_01 --episodes 20
uv run baba-viz frame tutorial_01 --out /tmp/tutorial_initial.png
uv run python -m baba_rlvr.training.grpo_train --smoke --env-url http://localhost:8000
```

## Hugging Face Space / OpenEnv Deployment

The repo is set up for a Docker Space:

- `openenv.yaml` declares the OpenEnv app target as `server:app` on port 8000.
- `Dockerfile` builds the environment, clones and patches the vendored Baba
  engine during image build, installs the package, and launches uvicorn.
- `.dockerignore` excludes checkpoints, trajectories, W&B logs, notebook
  caches, and other large local artifacts from the build context.

Recommended submission flow:

1. Create a dedicated Hugging Face Docker Space repository.
2. Push this codebase without local training artifacts.
3. Verify `/health`, `/docs`, and `/play` on the live Space.
4. Paste the final Space URL and public training-evidence links into the
   table at the top of this README.

If you are using the OpenEnv CLI, the repo now has the expected manifest for a
command like `openenv push --repo-id <user-or-org>/baba-rlvr`.

## Repository Layout

```text
src/baba_rlvr/
├── engine/        pyBaba adapter and world projection
├── server/        FastAPI OpenEnv server and browser play UI
├── reward/        Verifiable milestone-based reward tracker
├── training/      GRPO trainers, prompts, reward callable
├── pcg/           MAP-Elites generator and BFS solver
├── eval/          Random and base-vs-trained evaluation harnesses
├── viz/           PNG/GIF rendering and trajectory visualization
├── memory/        Verifier-gated scratchpad support
└── client/        Thin HTTP client for training/eval
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module map.

## License

MIT.