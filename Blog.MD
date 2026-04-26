# Baba Is Solved? Teaching a Language Model to Rewrite the Laws of Physics

*A submission to the Meta × OpenEnv Hackathon, India 2026*

---

**TL;DR:** We turned *Baba Is You* into an OpenEnv RLVR environment to study a specific question: can a language model keep a coherent world model when the rules of the world are themselves movable objects? In this game, the agent is not just solving a maze or walking to a goal. It has to reason about symbolic rules like `WALL IS STOP`, `FLAG IS WIN`, or `WATER IS SINK`, then decide when to obey them, when to rewrite them, and how those edits change the environment dynamics. Our environment exposes the state as a readable ASCII grid plus the active rule set, controllable entities, and win conditions, so the model sees the task in language rather than pixels. On top of that, we built verifiable rewards, mechanic-specific milestones, and a staged curriculum so training is grounded and measurable instead of hand-wavy. The result is a compact but unusually rich benchmark for planning, world modeling, and reasoning under shifting rules, where success depends on learning that the laws of physics are part of the puzzle.

There is a moment in the puzzle game *Baba Is You* that I find genuinely unsettling. You are standing in front of a wall. The wall is, by the rules of the world, impassable: `WALL IS STOP` is written plainly in the top corner of the screen, three word-tiles in a row, as obvious and absolute as a posted sign. And then it occurs to you: those tiles are *objects*. You can push them. You can walk up to the word `STOP`, shove it sideways, and watch the wall, that same wall, become something you can simply walk through.

I don't think I have ever encountered a more compact demonstration of what it means to reason about a world whose rules are themselves part of the world.

That is the problem we set out to build a training environment around. Not the puzzle game but the *capability*. Can a language model learn to plan not just *through* an environment but *about* it, maintaining a coherent world model even as the rules of that world are objects it can directly manipulate? We believe the answer is yes, and we built an OpenEnv-compliant RLVR environment to try to prove it.

---

## The Problem With Language Models and Moving Walls

Ask a current language model to navigate a static grid to a fixed goal and it will do reasonably well. Ask it to navigate a grid where the goal can be moved, the obstacles can be dissolved, and its own identity can be reassigned mid-episode, and something breaks.

This is, at its core, a failure of *world modeling*. A strong reasoner does not only track *where things are*. It tracks *what things mean*, *what rules apply*, and *how those rules can change*. Baba Is You, designed by Arvi Teikari as an artistic puzzle game, turns out to be an extraordinarily precise surgical instrument for probing exactly this capability in a learning agent.

The rules of the game are sentences. `BABA IS YOU` means Baba is the player-controlled entity. `FLAG IS WIN` means reaching the flag wins the level. `WALL IS STOP` means walls block movement. These sentences are formed by arranging physical word-tiles (nouns, the copula IS, properties) in horizontal or vertical lines on the grid. And because the tiles are physical, they can be pushed. The rules can be broken, rewritten, invented.

In one level, the path to the flag is blocked by water marked `WATER IS SINK`. The agent can drown. The obvious solution is to avoid the water. The elegant solution is to push `SINK` away from `WATER IS`, removing the lethal property entirely.

---

## Building the Laboratory

We wrapped the C++ `pyBaba` engine (the same battle-tested game core that powers the original) in a thin Python adapter and exposed it through a FastAPI server following the OpenEnv session contract. The server speaks the standard vocabulary: `/reset`, `/step`, `/state`, `/close`. The agent speaks the standard observation language: an ASCII grid, a list of active rules in `SUBJECT IS PROPERTY` form, the set of controllable entities, the set of win entities.

The deliberate choice to keep observations *symbolic and readable* is not accidental. The grid is rendered as ASCII tokens, not pixels. The rules are emitted as structured text. This matters because we want the language model to reason in the same medium it was pretrained on, language, rather than forcing a perceptual bottleneck at the input.

Each observation contains, in full:

- The grid in ASCII (what the world looks like)
- The active rule set (what the laws of physics currently are)
- Which entities you control (who you are)
- Which entities win the game (where you're going)
- How many steps remain (how long you have to think)

The agent returns a single JSON object: an action and a rationale. Up, down, left, right, wait. The rationale is for us; the engine ignores it. But the discipline of articulating *why* turns out to matter for the gradient; we return to this.

---

## The Verifier: A Reward Function That Cannot Be Fooled

Here is the central design challenge in reinforcement learning from verifiable rewards: the reward signal must be *true*. Not approximately true. Not true-under-normal-circumstances. True in the sense that no amount of adversarial cleverness by the agent can manufacture a high reward without actually solving the task.

We achieve this by building the reward tracker as a pure *observer of engine state*. It reads `PlayState`, `RuleManager`, and entity positions directly from the C++ game object. It never reads the agent's reasoning text. The agent could output `"I have won the game"` in its rationale on every step and accumulate exactly zero reward for it. The only path to reward is *doing the thing in the world*.

The reward structure is deliberately layered. A small per-step cost of `-0.01` per move bleeds out slowly, making infinite loops an economically losing proposition. Death costs `-2.0`. Invalid actions cost `-0.5`. These are background signals, the physiological noise of a functioning organism.

The interesting signals are the milestones: one-time rewards that fire at most once per episode, tied to specific verifiable events in the game engine.

| Milestone | Reward |
|---|---|
| Breaking a rule active at episode start | +1.0 |
| Creating a genuinely novel rule | +1.0 |
| Establishing a new win condition | +1.5 |
| Changing which entity is `YOU` | +2.0 |
| Neutralizing a kill effect | +1.0 |
| **Winning** | **+10.0** |

These milestones are curriculum without being curriculum. We are not telling the agent to break a rule. We are simply observing, with a flicker of positive signal, when it does. The agent must discover the relevance of rule-manipulation on its own. The reward is *verification*, not instruction.

Re-creating the same rule after breaking it earns nothing; we track rule signatures and block repeat exploitation. The loop-farming strategy that plagues naive milestone-based rewards is simply insolvent here.

And crowning everything: **+10.0** for winning. The terminal reward dwarfs every milestone combined, which means the agent can never get a high score by farming intermediate bonuses. Winning remains the dominant objective. Everything else is scaffolding toward it.

---

## A Curriculum Built on Geometric Progression

We designed twenty-four levels, organized into eight tiers of three, each tier introducing exactly one new mechanic. The progression is chosen to mirror how a thoughtful teacher would introduce the game to a student who had never seen it before.

| Tier | Focus | What the agent must learn |
|---|---|---|
| T1 | Movement | Can you reach the flag? |
| T2 | STOP walls | Can you find a path? |
| T3 | PUSH rocks | Can you clear the path? |
| T4 | Breaking STOP | Can you break a rule? |
| T5 | Water / SINK | Can you neutralize a hazard? |
| T6 | Creating WIN | Can you manufacture the win condition? |
| T7 | Schema drift | Can you adapt when you are no longer Baba? |
| T8 | OPEN / SHUT | Can you reason about key-door interactions? |

Within each tier, two levels are used for training and one (the `_eval` level) is held out. The held-out levels test generalization to novel instances of the same mechanic; they are the fair judges of whether the model learned the *concept* or merely memorized the specific configuration.

This structure answers a question that haunts almost every RL training effort: *how do we know the agent actually learned something?* The held-out comparison provides a baseline that is honest precisely because the agent never trained on it.

---

## Training: GRPO and the Group Baseline

We train with GRPO (Group Relative Policy Optimization) using Hugging Face TRL and Unsloth on a Qwen2.5 base. The choice is principled: GRPO requires no value network, which is significant when each environment rollout is an HTTP round-trip to a game engine running on a separate process. There is no critic to maintain, no separate value loss to balance. The group of K rollouts from the same prompt provides its own baseline.

The training loop is worth describing precisely because it is the clearest illustration of what makes this environment interesting for RLVR.

A prompt is constructed from a level's initial observation. The model generates K completions: different action choices, different rationales, different plans. Each completion is played out as an episode against the environment server: the first completion becomes the first action, subsequent actions are generated turn by turn against the updated game state, for up to forty turns. Each step accrues reward from the tracker. The discounted sum becomes a scalar, and the scalar enters the GRPO update as the rollout's return.

The group baseline normalizes: a return of +9.9 in a group whose mean is +5.0 receives a strongly positive advantage. A return of +0.6 in the same group is mildly penalized. The model learns not *did I win* but *did I win compared to what I would otherwise have done*. This relative framing is precisely what drives the kind of differential improvement we are after: not pushing the model to maximize an absolute score, but pushing it to consistently choose the better plan within the space of its own uncertainty.

**Training configuration:**

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Trainer: TRL `GRPOTrainer` with adaptive curriculum sampler
- Acceleration: Unsloth, 4-bit quantization, fp16
- LoRA: `r=16`, `alpha=32`
- Steps: 150 | Generations per prompt: 8
- Batch size: 4 | Grad accumulation: 2
- Learning rate: `5e-6` | Max turns per rollout: 30
- Curriculum advance threshold: 0.6 rolling win rate

---

## What We Are Actually Testing

Let us be direct about the scientific claim, because clarity here is what separates a genuine contribution from a demo.

We are not testing whether a language model can play Baba Is You. We are testing whether, through reinforcement learning on a carefully designed verifiable environment, a language model can acquire a *functional world model* for a domain where the rules are mutable.

The held-out evaluation tells us whether the trained model generalizes. The before/after comparison tells us whether something changed. The milestone distribution tells us *what* changed: whether the model began breaking rules more often, creating win conditions, or surviving schema drift. The reward curve tells us when.

And the one thing we are honest about in advance: we expect the model to still fail on the hardest levels. Tier 7 and Tier 8 require reasoning about identity change and key-door interactions that are genuinely compositional in a way that 150 training steps may not resolve. We flag this not as a limitation to hide but as the edge of the frontier we have actually found, which is, scientifically, the most interesting thing to report.

---

## Results

- **Before training**: the model frequently produced invalid moves or treated the puzzle as static navigation, moving toward the flag without accounting for blocking rules.
- **After training**: the model more consistently broke obstructive rules, created win conditions intentionally, and began using key/door and identity-switch mechanics on tiers it had seen during training.

We were not able to fully train across all generated levels within the available hackathon budget, so these results should be read as an encouraging partial study rather than a complete sweep. With more training time and compute, the natural next step is to expand coverage across the full generated set and report a broader, more comprehensive evaluation.

### Plots

![Mean rollout reward during training](assets/blog/W%26B%20Chart%204_26_2026,%204_01_29%20PM.png)

*Mean rollout reward over training, showing how the policy improves as it learns to manipulate rules rather than treating the levels as static navigation problems.*

![Total rollout wins during training](assets/blog/W%26B%20Chart%204_26_2026,%204_01_46%20PM.png)

*Total rollout wins during training, giving a more concrete view of when the model begins converting improved reasoning into actual solved episodes.*

---

## Why This Matters Beyond the Game

Real agents regularly encounter environments where the rules change mid-task. API schemas drift. Policies update. Systems switch modes. A model trained only on static tasks develops what we might call *brittle confidence*: it acts as though the world it sees is the world it will always see.

Baba Is You turns this into a controlled, inspectable, reproducible training problem. The rule changes are discrete, verifiable, and grounded in explicit symbolic state. The reward signal is tamper-proof. The curriculum is designed to escalate the difficulty of rule manipulation systematically.

We built a laboratory for the question: *what does it take for a language model to hold a world model in its head while the world rewrites itself around it?*

We don't have the full answer yet. But we have, we think, the right experiment.

---

## Links

- **Environment (Hugging Face Space):** [amannnnn/baba-rlvr](https://huggingface.co/spaces/amannnnn/baba-rlvr)
- **Training notebook (GitHub):** [notebooks/colab_grpo_demo.ipynb](https://github.com/amannnnn-7/babaissolved/blob/main/notebooks/colab_grpo_demo.ipynb)
- **Run script used for the main training run:** [scripts/run_qwen_a100_grpo.sh](https://github.com/amannnnn-7/babaissolved/blob/main/scripts/run_qwen_a100_grpo.sh)
- **W&B training dashboard:** [baba-rlvr W&B project](https://microsoft-research.wandb.io/t-apaliwal/baba-rlvr?nw=nwusertapaliwal)
- **Code:** [github.com/amannnnn-7/babaissolved](https://github.com/amannnnn-7/babaissolved)
