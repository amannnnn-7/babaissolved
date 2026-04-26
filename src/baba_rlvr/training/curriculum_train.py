"""Curriculum-aware GRPO training for the official Baba Is You level pack.

Key differences vs ``grpo_train.py``:

* Fixes the unsloth dtype bug we hit with Qwen3-4B + 4-bit + bf16 by loading
  in fp16 and using ``fp16=True`` in the trainer.
* Defaults to ``Qwen/Qwen2.5-1.5B-Instruct`` so the demo run completes inside
  ~30-60 minutes on a single A100 with visible reward improvement.
* Builds the training dataset from a tier-ordered cycle of ``levels/official/``
  training levels (eval levels held out — see ``training/curriculum.py``).
* Forces a *sequential* sampler so the model walks T1 → T8, producing clean
  step-vs-tier-vs-reward W&B curves.
* Logs per-call rollout metrics to W&B every reward call: scalar ``rollout/*``
  series for mean reward, win-rate, invalid-rate, mean steps, mean milestones,
  active tier, plus per-tier rolling win-rates ``tier/*/win_rate``.

Run it via ``scripts/run_qwen_a100_grpo.sh`` (which now starts the env server,
trains, then runs the base-vs-trained eval).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset

from ..server.env import BabaEnv
from .curriculum import TIERS, split
from .prompts import SYSTEM_PROMPT, build_prompt
from .reward_callable import episode_return


# ---------------------------------------------------------------------------
# Adaptive curriculum state
# ---------------------------------------------------------------------------


@dataclass
class CurriculumState:
    """Mutable curriculum controller shared between sampler and reward func.

    The trainer stays on ``active_tier`` until the rolling win-rate over the
    most recent ``min_episodes`` episodes meets ``advance_threshold``. Then it
    advances to the next tier (capped at the last tier).
    """

    tier_keys: list[str]
    advance_threshold: float = 0.6
    min_episodes: int = 32
    history_maxlen: int = 128
    active_idx: int = 0
    history: dict[str, deque] = field(default_factory=dict)
    advancements: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for t in self.tier_keys:
            self.history.setdefault(t, deque(maxlen=self.history_maxlen))

    @property
    def active_tier(self) -> str:
        return self.tier_keys[self.active_idx]

    def record(self, tier: str, won: bool) -> None:
        if tier in self.history:
            self.history[tier].append(int(won))

    def maybe_advance(self, episodes_seen: int) -> bool:
        if self.active_idx >= len(self.tier_keys) - 1:
            return False
        hist = self.history[self.active_tier]
        # Look at the most recent min_episodes on this tier only.
        recent = list(hist)[-self.min_episodes:]
        if len(recent) < self.min_episodes:
            return False
        win_rate = sum(recent) / len(recent)
        if win_rate >= self.advance_threshold:
            prev = self.active_tier
            self.active_idx += 1
            self.advancements.append(
                {
                    "from": prev,
                    "to": self.active_tier,
                    "episodes_seen": episodes_seen,
                    "win_rate": win_rate,
                }
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_curriculum_dataset(
    train_levels: list[str], tier_of: dict[str, str]
) -> tuple[Dataset, dict[str, list[int]]]:
    """One row per training level + a tier->row-indices map.

    The dataset is small (== len(train_levels)). The adaptive sampler reaches
    into ``tier_to_indices[active_tier]`` to pick which row to feed the model
    each step, so the dataset itself is just a static lookup table.
    """
    rows: list[dict[str, Any]] = []
    tier_to_indices: dict[str, list[int]] = {}
    for lid in train_levels:
        env = BabaEnv(level_id=lid)
        obs = env.reset()
        tier = tier_of[lid]
        tier_to_indices.setdefault(tier, []).append(len(rows))
        rows.append({"prompt": build_prompt(obs), "level_id": lid, "tier": tier})
    return Dataset.from_list(rows), tier_to_indices


# ---------------------------------------------------------------------------
# Reward function with rich W&B logging
# ---------------------------------------------------------------------------


def make_curriculum_reward_func(
    *,
    env_url: str,
    tier_of: dict[str, str],
    state: CurriculumState,
    max_turns: int,
    use_memory: bool,
    trajectory_log_dir: Path | None,
    log_to_wandb: bool,
    rolling_window: int = 64,
):
    log_path: Path | None = None
    if trajectory_log_dir:
        trajectory_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = trajectory_log_dir / "trajectories.jsonl"

    # Rolling per-tier win history (separate from CurriculumState.history,
    # which uses a different window for advancement decisions).
    tier_history: dict[str, deque] = {t: deque(maxlen=rolling_window) for t in TIERS}
    tier_returns: dict[str, deque] = {t: deque(maxlen=rolling_window) for t in TIERS}
    cumulative: dict[str, int] = {"calls": 0, "episodes": 0, "wins": 0}

    def reward_func(prompts, completions, **kwargs) -> list[float]:
        cumulative["calls"] += 1
        level_ids: list[str] = list(kwargs.get("level_id") or [])
        if not level_ids:
            level_ids = ["t1_01_baba_is_you"] * len(completions)
        rewards: list[float] = []
        wins = 0
        invalids = 0
        steps_total = 0
        milestones_total = 0
        active_tiers: list[str] = []
        records: list[dict[str, Any]] = []

        for prompt, comp, lid in zip(prompts, completions, level_ids, strict=False):
            text = comp if isinstance(comp, str) else comp[0]["content"]
            ret, diag = episode_return(
                env_url=env_url,
                level_id=lid,
                completion=text,
                max_turns=max_turns,
                use_memory=use_memory,
                initial_prompt=prompt,
            )
            rewards.append(float(ret))
            tier = tier_of.get(lid, "T?")
            active_tiers.append(tier)
            won = bool(diag["won"])
            if tier in tier_history:
                tier_history[tier].append(int(won))
                tier_returns[tier].append(float(ret))
            cumulative["episodes"] += 1
            cumulative["wins"] += int(won)
            wins += int(won)
            invalids += int(diag["invalid"])
            steps_total += int(diag["steps"])
            milestones_total += len(diag["milestones"])
            # Feed the adaptive curriculum its outcome signal.
            state.record(tier, won)
            records.append(
                {
                    "episode_id": f"{cumulative['calls']:06d}-{lid}-{int(time.time()*1e6)%1_000_000:06d}",
                    "timestamp": time.time(),
                    "level_id": lid,
                    "tier": tier,
                    "return": float(ret),
                    "won": won,
                    "steps": int(diag["steps"]),
                    "invalid": int(diag["invalid"]),
                    "n_actions_planned": int(diag["n_actions_planned"]),
                    "milestones": diag["milestones"],
                    "first_completion": text,
                }
            )

        n = max(1, len(rewards))
        # Try to advance the curriculum after the batch is scored.
        advanced = state.maybe_advance(cumulative["episodes"])
        if advanced:
            adv = state.advancements[-1]
            print(
                f"[curric] ADVANCE {adv['from']} -> {adv['to']} "
                f"(win_rate={adv['win_rate']:.0%} after {adv['episodes_seen']} eps)"
            )

        metrics: dict[str, Any] = {
            "rollout/reward_mean": sum(rewards) / n,
            "rollout/reward_max": max(rewards) if rewards else 0.0,
            "rollout/reward_min": min(rewards) if rewards else 0.0,
            "rollout/win_rate": wins / n,
            "rollout/invalid_per_episode": invalids / n,
            "rollout/mean_steps": steps_total / n,
            "rollout/mean_milestones": milestones_total / n,
            "rollout/episodes_total": cumulative["episodes"],
            "rollout/wins_total": cumulative["wins"],
            "rollout/cumulative_win_rate": cumulative["wins"] / max(1, cumulative["episodes"]),
            # Curriculum controller telemetry.
            "curriculum/active_tier_idx": state.active_idx,
            "curriculum/active_tier_advancements": len(state.advancements),
        }
        # Most-common tier in *this batch* (for verifying the sampler did its job).
        if active_tiers:
            top_tier = max(set(active_tiers), key=active_tiers.count)
            metrics["curriculum/batch_tier_idx"] = list(TIERS.keys()).index(top_tier)
            metrics["curriculum/batch_tier"] = top_tier
        # Per-tier rolling win-rates and returns.
        for t, hist in tier_history.items():
            if hist:
                metrics[f"tier/{t}/win_rate"] = sum(hist) / len(hist)
                metrics[f"tier/{t}/n"] = len(hist)
            if tier_returns[t]:
                rs = tier_returns[t]
                metrics[f"tier/{t}/return_mean"] = sum(rs) / len(rs)

        _write_records(log_path, records)
        if log_to_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({k: v for k, v in metrics.items() if not isinstance(v, str)}, commit=False)
            except ImportError:
                pass
        return rewards

    return reward_func


def _write_records(path: Path | None, records: list[dict[str, Any]]) -> None:
    if path is None or not records:
        return
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Trainer entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default=os.environ.get("BABA_ENV_URL", "http://localhost:8000"))
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="ckpt-baba-curriculum")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--advance-win-rate", type=float, default=0.6,
                   help="Advance to the next tier when rolling win-rate on the "
                        "active tier reaches this fraction.")
    p.add_argument("--min-episodes-per-tier", type=int, default=32,
                   help="Require at least this many episodes on the active "
                        "tier before considering advancement.")
    p.add_argument("--use-memory", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "baba-rlvr"))
    p.add_argument("--wandb-run-name", default=os.environ.get("WANDB_NAME", "qwen2.5-baba-curriculum"))
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--trajectory-log-dir", type=Path,
                   default=Path("runs/qwen-curriculum-trajectories"))
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--smoke", action="store_true",
                   help="Skip model load; check the dataset + reward pipeline.")
    args = p.parse_args()

    print(f"[curric] system prompt:\n{SYSTEM_PROMPT}\n")

    sp = split()
    print(f"[curric] train levels ({len(sp.train)}): {sp.train}")
    print(f"[curric] eval  levels ({len(sp.eval)}): {sp.eval}")

    ds, tier_to_indices = build_curriculum_dataset(sp.train, sp.tier_of)
    tier_keys = list(TIERS.keys())
    # Drop tiers with no train levels (shouldn't happen, but be defensive).
    tier_keys = [t for t in tier_keys if tier_to_indices.get(t)]
    state = CurriculumState(
        tier_keys=tier_keys,
        advance_threshold=args.advance_win_rate,
        min_episodes=args.min_episodes_per_tier,
    )
    print(f"[curric] dataset rows: {len(ds)}; tiers: {tier_keys}; "
          f"start tier: {state.active_tier}; "
          f"advance @ win_rate>={args.advance_win_rate} over "
          f"{args.min_episodes_per_tier} eps")

    if args.smoke:
        reward_fn = make_curriculum_reward_func(
            env_url=args.env_url, tier_of=sp.tier_of, state=state,
            max_turns=args.max_turns, use_memory=args.use_memory,
            trajectory_log_dir=args.trajectory_log_dir,
            log_to_wandb=False,
        )
        smoke_completion = "REASONING: just walk right.\nPLAN:\nright\nright\nright\n"
        # Pull rows from the active tier to mirror what the live sampler does.
        idx = tier_to_indices[state.active_tier][:4]
        if len(idx) < 4:
            idx = (idx * 4)[:4]
        rewards = reward_fn(
            [ds[i]["prompt"] for i in idx],
            [smoke_completion] * 4,
            level_id=[ds[i]["level_id"] for i in idx],
        )
        print(f"[curric] smoke rewards: {rewards}")
        return

    # ---- Heavy imports only outside smoke mode ---------------------------
    FastLanguageModel = importlib.import_module("unsloth").FastLanguageModel
    import random as _random
    import torch
    from trl import GRPOConfig, GRPOTrainer
    from torch.utils.data import Sampler

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)

    # Load fp16 to avoid the Half/BFloat16 mismatch in unsloth's LoRA QKV
    # kernel that crashed the previous Qwen3-4B + bf16 run before step 1.
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_prompt_length + args.max_completion_length,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    @torch_no_grad
    def generate(prompt: str) -> str:
        """Plain HF generate. Unused by GRPO (it samples internally) but kept\n        for ad-hoc debugging from a REPL."""
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
        return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    _ = generate  # silence unused warnings; reachable via locals() in a REPL.

    reward_fn = make_curriculum_reward_func(
        env_url=args.env_url, tier_of=sp.tier_of, state=state,
        max_turns=args.max_turns, use_memory=args.use_memory,
        trajectory_log_dir=args.trajectory_log_dir,
        log_to_wandb=not args.no_wandb,
    )

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=1,
        max_steps=args.steps,
        logging_steps=1,
        save_strategy="no",
        fp16=True,
        bf16=False,
        beta=0.04,
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=args.wandb_run_name,
    )

    # Adaptive sampler: every index it yields is randomly drawn from the
    # *current* active tier's level pool. The reward function mutates
    # ``state.active_idx`` once a tier is mastered, so the very next batch
    # automatically samples from the next tier -- no dataset rebuild needed.
    rng = _random.Random(0)
    total_samples_needed = (
        args.steps * args.batch_size * args.grad_accum * args.num_generations
    ) + args.batch_size  # small headroom for trainer prefetch
    print(f"[curric] adaptive sampler will draw up to {total_samples_needed} indices")

    class _AdaptiveCurriculumSampler(Sampler):
        def __init__(self, length: int) -> None:
            self._length = length

        def __len__(self) -> int:
            return self._length

        def __iter__(self):
            for _ in range(self._length):
                pool = tier_to_indices[state.active_tier]
                yield rng.choice(pool)

    class _CurriculumGRPOTrainer(GRPOTrainer):
        def _get_train_sampler(self, *a, **kw):  # noqa: D401
            return _AdaptiveCurriculumSampler(total_samples_needed)

    trainer = _CurriculumGRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"[curric] saved adapter to {args.out}")
    print(f"[curric] final tier reached: {state.active_tier} "
          f"(idx {state.active_idx}/{len(state.tier_keys) - 1}); "
          f"advancements: {len(state.advancements)}")
    for adv in state.advancements:
        print(f"[curric]   {adv['from']} -> {adv['to']} @ ep={adv['episodes_seen']} "
              f"win_rate={adv['win_rate']:.0%}")


def torch_no_grad(fn):
    def wrapper(*a, **kw):
        import torch

        with torch.no_grad():
            return fn(*a, **kw)

    return wrapper


if __name__ == "__main__":
    main()
