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
        --model unsloth/Qwen2.5-3B-Instruct-bnb-4bit \
        --steps 200 \
        --use-memory false
"""

from __future__ import annotations

import argparse
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
    if curriculum_path and curriculum_path.exists():
        with curriculum_path.open("rb") as f:
            archive = pickle.load(f)
        for elite in archive.values():
            register_level(elite.level_id, elite.spec)
        ids = list(archive.keys())
        from random import Random

        rng = Random(0)
        sampled = [archive[rng.choice(ids)].level_id for _ in range(n)]
    else:
        templates = sorted(LEVEL_REGISTRY.keys())
        sampled = [templates[i % len(templates)] for i in range(n)]

    from .prompts import build_prompt as _bp  # noqa: F401  (kept for clarity)
    from ..server.env import BabaEnv

    for lid in sampled:
        env = BabaEnv(level_id=lid)
        obs = env.reset()
        rows.append({"prompt": build_prompt(obs), "level_id": lid})
    return Dataset.from_list(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default=os.environ.get("BABA_ENV_URL", "http://localhost:8000"))
    p.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    p.add_argument("--curriculum", type=Path, default=Path("levels/archive.pkl"))
    p.add_argument("--out", default="ckpt-baba-grpo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--dataset-size", type=int, default=512)
    p.add_argument("--use-memory", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--smoke", action="store_true", help="Skip model load; test pipeline only.")
    args = p.parse_args()

    print(f"[grpo] system prompt:\n{SYSTEM_PROMPT}\n")
    ds = _build_dataset(args.curriculum, args.dataset_size)
    print(f"[grpo] dataset rows: {len(ds)}; first level: {ds[0]['level_id']}")

    if args.smoke:
        print("[grpo] smoke mode: dummy generator returns 'right'.")
        gen = lambda _prompt: '{"action":"right","rationale":"smoke"}'  # noqa: E731
        reward_fn = make_reward_func(args.env_url, gen, max_turns=args.max_turns,
                                     use_memory=args.use_memory)
        rewards = reward_fn(ds["prompt"][:2], [gen("")] * 2, level_id=ds["level_id"][:2])
        print(f"[grpo] smoke rewards: {rewards}")
        return

    # ---- Heavy imports happen only outside smoke mode --------------------
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel  # type: ignore

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=4096, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    @torch_no_grad
    def generate(prompt: str) -> str:
        import torch

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
        args.env_url, generate, max_turns=args.max_turns, use_memory=args.use_memory
    )

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=5e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=args.num_generations,
        max_prompt_length=2048,
        max_completion_length=48,
        num_train_epochs=1,
        max_steps=args.steps,
        logging_steps=1,
        bf16=True,
        beta=0.04,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
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
