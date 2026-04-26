"""Minimal GRPO training script — Colab-friendly.

Run on a single GPU (Colab T4/L4/A100). For CPU sanity testing you can pass
``--smoke`` to skip the model load and verify the data pipeline + reward
callable end-to-end against a running env server.

Usage
-----
    # Terminal 1: env server
    uv run baba-server

    # Terminal 2: training (needs `train` + `train-unsloth` extras)
    uv run python -m baba_rlvr.training.grpo_train \
        --env-url http://localhost:8000 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --steps 200 \
        --use-memory false
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
from pathlib import Path

from ..levels.loader import LEVEL_REGISTRY, register_level
from .prompts import SYSTEM_PROMPT, build_prompt
from .reward_callable import make_reward_func


def _build_dataset(curriculum_path: Path | None, n: int):
    """Build a HF Dataset of {prompt, level_id} rows.

    If a MAP-Elites archive is supplied we register every elite as a level
    and sample uniformly. Otherwise we cycle through the built-in templates.
    """
    from datasets import Dataset  # imported lazily; only needed at training time

    rows = []
    if curriculum_path and curriculum_path.exists() and curriculum_path.is_dir():
        for map_path in sorted(curriculum_path.glob("*.txt")):
            level_id = map_path.stem
            register_level(level_id, {"map_path": str(map_path.resolve()), "max_steps": 80})
        generated = [p.stem for p in sorted(curriculum_path.glob("*.txt"))]
        if not generated:
            raise ValueError(f"curriculum directory has no .txt maps: {curriculum_path}")
        sampled = [generated[i % len(generated)] for i in range(n)]
    elif curriculum_path and curriculum_path.exists():
        with curriculum_path.open("rb") as f:
            archive = pickle.load(f)
        elites = list(archive.values()) if isinstance(archive, dict) else list(archive)
        for elite in elites:
            register_level(elite.level_id, elite.spec)
        from random import Random

        rng = Random(0)
        sampled = [rng.choice(elites).level_id for _ in range(n)]
    else:
        templates = sorted(LEVEL_REGISTRY.keys())
        sampled = [templates[i % len(templates)] for i in range(n)]

    from ..server.env import BabaEnv

    for lid in sampled:
        env = BabaEnv(level_id=lid)
        obs = env.reset()
        rows.append({"prompt": build_prompt(obs), "level_id": lid})
    return Dataset.from_list(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default=os.environ.get("BABA_ENV_URL", "http://localhost:8000"))
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--curriculum", type=Path, default=Path("levels/archive.pkl"))
    p.add_argument("--out", default="ckpt-baba-grpo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--dataset-size", type=int, default=512)
    p.add_argument("--use-memory", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "baba-rlvr"))
    p.add_argument("--wandb-run-name", default=os.environ.get("WANDB_NAME", "qwen3-baba-grpo-a100"))
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    p.add_argument("--trajectory-log-dir", type=Path, default=Path("runs/trajectories"))
    p.add_argument(
        "--no-wandb-trajectories",
        action="store_true",
        help="Do not log rollout trajectory tables to W&B.",
    )
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--smoke", action="store_true", help="Skip model load; test pipeline only.")
    args = p.parse_args()

    print(f"[grpo] system prompt:\n{SYSTEM_PROMPT}\n")
    ds = _build_dataset(args.curriculum, args.dataset_size)
    print(f"[grpo] dataset rows: {len(ds)}; first level: {ds[0]['level_id']}")

    if args.smoke:
        print("[grpo] smoke mode: emitting a dummy plan completion.")
        reward_fn = make_reward_func(
            args.env_url,
            max_turns=args.max_turns,
            use_memory=args.use_memory,
            trajectory_log_dir=args.trajectory_log_dir,
            log_to_wandb=not args.no_wandb and not args.no_wandb_trajectories,
        )
        smoke_completion = "REASONING: smoke.\nPLAN:\nright\nright\n"
        rewards = reward_fn(
            ds["prompt"][:2],
            [smoke_completion] * 2,
            level_id=ds["level_id"][:2],
        )
        print(f"[grpo] smoke rewards: {rewards}")
        print(f"[grpo] trajectories: {args.trajectory_log_dir / 'trajectories.jsonl'}")
        return

    # ---- Heavy imports happen only outside smoke mode --------------------
    FastLanguageModel = importlib.import_module("unsloth").FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=4096, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    @torch_no_grad
    def generate(prompt: str) -> str:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return text

    reward_fn = make_reward_func(
        args.env_url,
        max_turns=args.max_turns,
        use_memory=args.use_memory,
        trajectory_log_dir=args.trajectory_log_dir,
        log_to_wandb=not args.no_wandb and not args.no_wandb_trajectories,
    )

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=2048,
        max_completion_length=48,
        num_train_epochs=1,
        max_steps=args.steps,
        logging_steps=1,
        bf16=True,
        beta=0.04,
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=args.wandb_run_name,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.out)


def torch_no_grad(fn):
    """Lazy wrapper so we don't import torch at module import time."""
    def wrapper(*a, **kw):
        import torch

        with torch.no_grad():
            return fn(*a, **kw)

    return wrapper


if __name__ == "__main__":
    main()
