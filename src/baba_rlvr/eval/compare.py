"""Base-vs-trained evaluation harness on the official level pack.

For each split (train levels and held-out eval levels), runs N rollouts per
level with each model and emits:

  * a per-level table written to JSONL (and W&B if available),
  * a tier-aggregated summary printed to stdout,
  * an optional ``runs/eval/comparison.json`` for the deck.

Usage::

    # Local (env server already running):
    uv run python -m baba_rlvr.eval.compare \\
        --base Qwen/Qwen2.5-1.5B-Instruct \\
        --trained ckpt-baba-curriculum \\
        --episodes-per-level 5

The trained checkpoint must be a PEFT/LoRA adapter on top of the base model;
we load it with ``PeftModel.from_pretrained`` after the base is loaded by
unsloth's ``FastLanguageModel``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..training.curriculum import TIERS, split
from ..training.reward_callable import episode_return


def _format_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def _evaluate_one(
    *,
    label: str,
    generate,
    env_url: str,
    levels: list[str],
    tier_of: dict[str, str],
    episodes_per_level: int,
    max_turns: int,
) -> list[dict[str, Any]]:
    """Run rollouts for every (level, episode) pair and return per-episode rows."""
    rows: list[dict[str, Any]] = []
    for lid in levels:
        for ep in range(episodes_per_level):
            # Plan-mode: the model emits a full plan in one completion; the
            # reward helper parses it and steps the env action-by-action.
            # We re-build the prompt once from the env reset to match training.
            import httpx
            with httpx.Client(timeout=60, base_url=env_url) as cx:
                r = cx.post("/reset", json={"level_id": lid, "use_memory": False}).json()
                cx.post(f"/close/{r['session_id']}")
            from ..server.schemas import BabaObservation
            from ..training.prompts import build_prompt
            obs = BabaObservation(**r["observation"])
            prompt = build_prompt(obs)
            completion = generate(prompt)
            ret, diag = episode_return(
                env_url=env_url,
                level_id=lid,
                completion=completion,
                max_turns=max_turns,
                use_memory=False,
                initial_prompt=prompt,
            )
            rows.append(
                {
                    "model": label,
                    "level_id": lid,
                    "tier": tier_of.get(lid, "T?"),
                    "episode": ep,
                    "return": float(ret),
                    "won": bool(diag["won"]),
                    "steps": int(diag["steps"]),
                    "invalid": int(diag["invalid"]),
                    "milestones": diag["milestones"],
                }
            )
            print(
                f"  [{label}] {lid:30s} ep={ep} return={ret:+.2f} won={diag['won']} "
                f"steps={diag['steps']} invalid={diag['invalid']}"
            )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_level: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_tier: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_level[(r["model"], r["level_id"])].append(r)
        by_tier[(r["model"], r["tier"])].append(r)
        by_model[r["model"]].append(r)

    def _stats(rs):
        n = max(1, len(rs))
        return {
            "n": len(rs),
            "win_rate": sum(int(r["won"]) for r in rs) / n,
            "return_mean": sum(r["return"] for r in rs) / n,
            "steps_mean": sum(r["steps"] for r in rs) / n,
        }

    return {
        "per_level": {f"{m}|{l}": _stats(rs) for (m, l), rs in by_level.items()},
        "per_tier": {f"{m}|{t}": _stats(rs) for (m, t), rs in by_tier.items()},
        "per_model": {m: _stats(rs) for m, rs in by_model.items()},
    }


def _build_generate(model_name: str, adapter_path: str | None, max_new_tokens: int):
    """Return a ``generate(prompt)->str`` closure backed by transformers/peft."""
    import torch

    FastLanguageModel = importlib.import_module("unsloth").FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)

    def generate(prompt: str) -> str:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return generate, model, tok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default=os.environ.get("BABA_ENV_URL", "http://localhost:8000"))
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--trained", required=True, help="Path to trained LoRA adapter.")
    p.add_argument("--episodes-per-level", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--out", type=Path, default=Path("runs/eval"))
    p.add_argument("--wandb-run-name", default=os.environ.get("WANDB_NAME", "baba-eval-base-vs-trained"))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "baba-rlvr"))
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    sp = split()
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "rollouts.jsonl"
    summary_path = args.out / "comparison.json"

    if not args.no_wandb:
        try:
            import wandb

            os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, reinit=True)
        except ImportError:
            pass

    all_rows: list[dict[str, Any]] = []

    for label, adapter in [("base", None), ("trained", args.trained)]:
        print(f"\n=== Evaluating {label} ({args.base}{' + ' + adapter if adapter else ''}) ===")
        gen, model, tok = _build_generate(args.base, adapter, args.max_new_tokens)
        print("\n-- TRAIN levels --")
        all_rows += _evaluate_one(
            label=label, generate=gen, env_url=args.env_url,
            levels=sp.train, tier_of=sp.tier_of,
            episodes_per_level=args.episodes_per_level, max_turns=args.max_turns,
        )
        print("\n-- HELD-OUT (eval) levels --")
        all_rows += _evaluate_one(
            label=label, generate=gen, env_url=args.env_url,
            levels=sp.eval, tier_of=sp.tier_of,
            episodes_per_level=args.episodes_per_level, max_turns=args.max_turns,
        )
        # Free GPU memory before loading the next model.
        del model, tok, gen
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    with rows_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    summary = _aggregate(all_rows)
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2))

    print("\n=== Per-tier summary (HELD-OUT eval levels only) ===")
    eval_set = set(sp.eval)
    eval_rows = [r for r in all_rows if r["level_id"] in eval_set]
    tier_table: list[dict[str, Any]] = []
    by_tier: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in eval_rows:
        by_tier[(r["model"], r["tier"])].append(r)
    for t in TIERS:
        for label in ("base", "trained"):
            rs = by_tier.get((label, t), [])
            n = max(1, len(rs))
            tier_table.append({
                "tier": t,
                "model": label,
                "n": len(rs),
                "win_rate": f"{sum(int(r['won']) for r in rs)/n:.0%}",
                "return": f"{sum(r['return'] for r in rs)/n:+.2f}",
            })
    print(_format_table(tier_table, ["tier", "model", "n", "win_rate", "return"]))

    print("\n=== Overall (train + eval combined) ===")
    overall = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_model[r["model"]].append(r)
    for m, rs in by_model.items():
        n = max(1, len(rs))
        overall.append({
            "model": m,
            "n": len(rs),
            "win_rate": f"{sum(int(r['won']) for r in rs)/n:.0%}",
            "return": f"{sum(r['return'] for r in rs)/n:+.2f}",
        })
    print(_format_table(overall, ["model", "n", "win_rate", "return"]))

    if not args.no_wandb:
        try:
            import wandb

            if wandb.run is not None:
                table = wandb.Table(
                    columns=["model", "split", "tier", "level_id", "episode",
                             "return", "won", "steps", "invalid"]
                )
                eval_set = set(sp.eval)
                for r in all_rows:
                    table.add_data(
                        r["model"],
                        "eval" if r["level_id"] in eval_set else "train",
                        r["tier"], r["level_id"], r["episode"],
                        r["return"], r["won"], r["steps"], r["invalid"],
                    )
                wandb.log({"eval/rollouts": table})
                # Scalar comparison metrics for plotting.
                for m, rs in by_model.items():
                    n = max(1, len(rs))
                    wandb.summary[f"eval/{m}/win_rate_all"] = sum(int(r['won']) for r in rs)/n
                    wandb.summary[f"eval/{m}/return_all"] = sum(r['return'] for r in rs)/n
                eval_only = [r for r in all_rows if r["level_id"] in eval_set]
                for m in ("base", "trained"):
                    rs = [r for r in eval_only if r["model"] == m]
                    if rs:
                        n = len(rs)
                        wandb.summary[f"eval/{m}/heldout_win_rate"] = sum(int(r['won']) for r in rs)/n
                        wandb.summary[f"eval/{m}/heldout_return"] = sum(r['return'] for r in rs)/n
                wandb.finish()
        except ImportError:
            pass

    print(f"\nWrote per-episode rollouts to {rows_path}")
    print(f"Wrote summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
