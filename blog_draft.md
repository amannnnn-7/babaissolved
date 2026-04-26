# Baba Is You for RLVR

Draft for the Hugging Face mini-blog or the narrative script for a short video.
This is intentionally detailed so it can be cut down later.

## Working Title Options

1. Teaching an LLM to Rewrite the Laws of Physics in Baba Is You
2. Training Language Models on Rule-Changing Puzzles with OpenEnv
3. Baba Is You as an RLVR Environment for Long-Horizon Rule Manipulation

## One-Paragraph Summary

We built an OpenEnv environment based on Baba Is You, the puzzle game where
the rules of the world are themselves objects inside the world. Instead of
only learning navigation, an agent has to learn to reason about and manipulate
sentences like `BABA IS YOU`, `FLAG IS WIN`, or `WALL IS STOP` to solve a
puzzle. That makes the task interesting for RLVR because rewards can be tied
to verifier-readable world events such as breaking a rule, creating a new
rule, changing which object is controllable, and finally reaching a win state.
We train the environment end-to-end with GRPO using TRL + Unsloth and expose
it through the OpenEnv HTTP contract so the training loop talks to a real
interactive environment instead of a static dataset.

## Problem Statement

Current LLM agents are decent at one-step decisions but much less reliable
when they need to maintain an internal world model while the rules of that
world are changing. Baba Is You is a natural stress test for that capability.

The agent has to answer questions like:

- Which rules are active right now?
- Which text blocks can I move to change those rules?
- If I break one rule, what becomes traversable or controllable next?
- Is it better to navigate directly to the flag, or first rewrite the win
  condition so a different object becomes the target?

This makes the environment relevant to the OpenEnv hackathon because it is not
just another gridworld. The task is about compositional reasoning over an
explicit, mutable rule system.

## The Game, Briefly

Baba Is You is a grid puzzle game where text has direct semantic force.
Placing the words `BABA`, `IS`, and `YOU` in a line means Baba becomes the
player-controlled entity. Placing `FLAG IS WIN` makes the flag the goal.
Placing `WALL IS STOP` makes walls impassable.

The key twist is that these words are also physical blocks on the board. The
agent can often solve a puzzle only by pushing words around and thereby
changing the rules of the environment.

That gives us a compact environment with:

- long-horizon plans,
- delayed reward,
- explicit symbolic state,
- changing action affordances,
- clear visual demos for judges.

## Environment Design

### Interface

The environment is served through FastAPI and follows the OpenEnv session
pattern:

- `POST /reset`
- `POST /step/{session_id}`
- `GET /state/{session_id}`
- `POST /close/{session_id}`
- `GET /levels`
- `GET /health`

For demos, we also serve:

- `GET /play` for manual interactive play,
- `GET /play/frame/{session_id}.png` for frame rendering,
- `GET /play/solve/{level_id}` for a BFS solution trace.

### Observation Space

Each observation includes:

- `grid_ascii`: a readable grid snapshot,
- `grid_tokens`: tokenized board contents,
- `active_rules`: the currently active `SUBJECT IS PROPERTY` facts,
- `you_entities`: which entities are currently controllable,
- `win_entities`: which entities currently satisfy `WIN`,
- `step_count` and `max_steps`,
- `level_id`,
- `memory_excerpt` when verifier-gated memory is enabled.

### Action Space

The low-level environment supports `up`, `down`, `left`, `right`, and `wait`.
For GRPO training, the model emits a short natural-language reasoning prefix
followed by a `PLAN:` section with one action per line.

This plan-mode protocol matters for RLVR because the full plan receives the
episode-level return, letting GRPO optimize sequences of actions rather than a
single token-level choice.

## Reward Design

The reward function is the verifier in the RLVR story. It reads structured
engine state only, never the model's free-form reasoning text, which makes the
reward robust against prompt hacking.

### Base Shaping

- win: `+10.0`
- death: `-2.0`
- invalid action: `-0.5`
- per-step cost: `-0.01`

### One-Time Milestone Rewards

- `first_rule_break`: `+1.0`
- `first_rule_make`: `+1.0`
- `self_redefine`: `+2.0`
- `win_condition_made`: `+1.5`
- `neutralized_kill`: `+1.0`

### Why These Rewards Work

The milestone rewards give dense signal for the kinds of strategic moves we
actually want the agent to discover:

- breaking an obstructive rule,
- creating a useful new rule,
- intentionally changing identity (`IS YOU`),
- manufacturing a win condition,
- removing a lethal interaction.

The terminal win reward is still dominant, so the agent cannot get a high
score just by farming intermediate milestones. We also block repeat milestone
fires and repeated rule-farming within the same episode.

## Engine Design

We initially had a pure-Python engine, but for correctness we pivoted to a
thin wrapper over the C++ `pyBaba` runtime from `utilForever/baba-is-auto`.
That let us keep the environment surface in Python while relying on a more
battle-tested game core.

We patch the vendored bindings and a small amount of gameplay logic to support
the mechanics needed for the hackathon pack, including:

- `AND` rule expansion,
- `SINK`,
- `OPEN` / `SHUT`,
- `MOVE`,
- `FLOAT`,
- visible water, ice, jelly, crab, love, and related word tiles.

This choice improved reliability while preserving a clean `World` API for the
server, reward tracker, solver, visualizer, and trainers.

## Level Design

We use three level sources.

### 1. Hand-Built RLVR Templates

The custom templates in `levels/templates/` are small, readable, and designed
to expose specific mechanics quickly during debugging and demos.

### 2. Official 24-Level Curriculum

The main training curriculum lives in `levels/official/` and is organized into
8 tiers with 3 levels each. In each tier, 2 levels are used for training and
1 level ending in `_eval` is held out for evaluation.

The tiers are:

1. Movement
2. STOP walls
3. PUSH rocks
4. Break a STOP rule
5. Water / SINK
6. Form a WIN rule
7. Schema drift (`IS YOU` changes)
8. OPEN / SHUT (key / door)

This ordering gives the model a progression from short navigation tasks to
more interesting rule manipulation and identity changes.

### 3. Procedural Curriculum via MAP-Elites

We also support solver-checked procedural generation. MAP-Elites mutates
grammar-preserving map layouts, verifies solvability with BFS, and writes a
diverse set of levels under `levels/_generated/`.

This is useful for expanding the task distribution beyond a small fixed pack.

## Training Recipe

There are two training entry points in the repo.

### Colab-Friendly Minimal GRPO Script

File: `src/baba_rlvr/training/grpo_train.py`

Default recipe:

- model: `Qwen/Qwen3-4B-Instruct-2507`
- trainer: TRL `GRPOTrainer`
- acceleration: Unsloth
- quantization: 4-bit
- LoRA: `r=32`, `alpha=64`
- steps: `200`
- generations per prompt: `8`
- batch size: `4`
- grad accumulation: `2`
- learning rate: `5e-6`
- max prompt length: `2048`
- max completion length: `48`
- max turns in environment rollout: `40`
- dataset size: `512`

This is the easiest path to show a re-runnable notebook demo.

### Main Hackathon A100 Curriculum Run

Launcher: `scripts/run_qwen_a100_grpo.sh`

Trainer backend: `src/baba_rlvr/training/curriculum_train.py`

Default recipe:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- trainer: TRL `GRPOTrainer` with an adaptive curriculum sampler
- acceleration: Unsloth
- quantization: 4-bit
- precision: fp16
- LoRA: `r=16`, `alpha=32`
- steps: `150`
- generations per prompt: `8`
- batch size: `4`
- grad accumulation: `2`
- learning rate: `5e-6`
- max turns: `30`
- max prompt length: `2048`
- max completion length: `2048`
- curriculum advance threshold: `0.6` rolling win rate
- minimum episodes per tier before advance: `32`

Why this recipe:

- the 1.5B model is small enough to train quickly on a single A100,
- the curriculum makes reward improvement easier to observe,
- the fp16 setup avoids the bf16 dtype mismatch that previously affected a
  Qwen3 4B run,
- held-out eval levels make the before/after comparison more credible.

### Logged Metrics

The training loop logs:

- trainer metrics such as loss and reward,
- rollout win rate,
- mean reward,
- mean steps,
- invalid actions per episode,
- cumulative win rate,
- active curriculum tier,
- rolling per-tier win rate and return,
- trajectory records for qualitative inspection.

## Evaluation Plan

The evaluation harness compares the base model and trained adapter on both the
training subset and the held-out eval levels.

For the final post, include:

1. base vs trained held-out win rate,
2. reward curve over training steps,
3. loss curve over training steps,
4. a few qualitative rollouts showing behavior change,
5. one sentence explaining where the model still fails.

## Results Section Template

Replace this section after the current run finishes.

Suggested structure:

### Quantitative Results

- Base held-out win rate: `TBD`
- Trained held-out win rate: `TBD`
- Best training win rate observed: `TBD`
- Final average rollout reward: `TBD`

### Qualitative Results

- Before training, the model often produced invalid moves or treated the game
  like static navigation.
- After training, the model more often broke obstructive rules, created win
  conditions intentionally, and used key/door or identity-switch mechanics.

### Plots To Embed

- reward vs training step,
- loss vs training step,
- held-out comparison chart,
- one rollout strip or GIF.

## Why This Matters

This environment is a compact testbed for a broader capability: can a language
model maintain a world model while the underlying rules of that world change?

That is relevant beyond games. Real agents regularly face situations where the
effective contract changes mid-task: policies update, tool schemas drift,
permissions change, or a system switches modes. Baba Is You turns that into a
clear, inspectable RLVR problem with a strong verifier and a good visual demo.

## Suggested 90-Second Video Outline

1. Start with one sentence: this is Baba Is You, but used as an RL training
   environment for language models.
2. Show a level where `WALL IS STOP` blocks progress.
3. Explain that the agent can move text to change the rules of the world.
4. Show the browser demo or a rendered rollout where the rule changes.
5. Explain the reward function in one sentence: reward is tied to verifiable
   world changes and actual success, not free-form text.
6. Show reward and loss curves from the real run.
7. End with base-vs-trained behavior on a held-out level.

## Final Editing Notes

Before publishing the final blog or recording the video:

1. Replace all `TBD` values with concrete numbers.
2. Add public links to the live Space, notebook, and W&B run.
3. Keep the first two paragraphs non-technical enough for judges skimming in 3 minutes.
4. Keep one short section on what still fails; it makes the submission more credible.