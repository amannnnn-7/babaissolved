"""GRPO reward callable: executes a *plan* against the env server.

Plan-mode protocol: TRL's GRPOTrainer feeds (prompt, completion) where the
completion is a multi-line plan emitted by the model. We parse the plan into
a sequence of actions, step the env action-by-action, and return the
discounted trajectory return as the scalar GRPO reward. Because all action
tokens live inside the trainable completion, every action token receives the
same trajectory-level advantage -- credit assignment now covers the whole
plan, not just the first action.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from ..server.schemas import ActionType, BabaAction
from .prompts import parse_plan


def episode_return(
    env_url: str,
    level_id: str,
    completion: str,
    *,
    max_turns: int = 40,
    gamma: float = 0.99,
    use_memory: bool = False,
    initial_prompt: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Execute a plan completion against the env; return discounted return + diag."""
    actions, invalid_lines = parse_plan(completion, max_actions=max_turns)
    diag: dict[str, Any] = {
        "won": False,
        "steps": 0,
        "milestones": [],
        "invalid": invalid_lines,
        "n_actions_planned": len(actions),
        "trajectory": [],
        "initial_prompt": initial_prompt,
        "completion": completion,
    }
    if not actions:
        # No parseable moves at all: charge invalid-action penalty for one WAIT
        # so the reward is well-defined and unambiguously bad.
        actions = [ActionType.WAIT]
        diag["invalid"] += 1

    with httpx.Client(timeout=60, base_url=env_url) as cx:
        r = cx.post("/reset", json={"level_id": level_id, "use_memory": use_memory}).json()
        sid = r["session_id"]
        ret = 0.0
        for t, action_kind in enumerate(actions):
            if t >= max_turns:
                break
            resp = cx.post(
                f"/step/{sid}",
                json=BabaAction(action=action_kind).model_dump(),
            ).json()
            step_reward = resp["reward"]
            ret += (gamma ** t) * step_reward
            for m, _ in resp["info"].get("milestones", []):
                if m not in diag["milestones"]:
                    diag["milestones"].append(m)
            diag["steps"] = t + 1
            diag["trajectory"].append(
                {
                    "turn": t,
                    "parsed_action": action_kind.value,
                    "reward": step_reward,
                    "done": resp["done"],
                    "info": resp["info"],
                }
            )
            if resp["done"]:
                diag["won"] = resp["info"].get("won", False)
                break
        cx.post(f"/close/{sid}")
    return ret, diag


def make_reward_func(
    env_url: str,
    *,
    max_turns: int = 40,
    use_memory: bool = False,
    trajectory_log_dir: Path | str | None = None,
    log_to_wandb: bool = True,
) -> Callable[..., list[float]]:
    """Factory that returns a TRL-compatible reward function.

    TRL's GRPOTrainer calls reward_funcs(prompts, completions, **kwargs) and
    expects a list[float] of length len(completions). We additionally expect
    a `level_id` column from the dataset, surfaced in **kwargs.
    """

    log_path: Path | None = None
    if trajectory_log_dir:
        log_dir = Path(trajectory_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "trajectories.jsonl"

    def reward_func(prompts, completions, **kwargs) -> list[float]:
        level_ids = kwargs.get("level_id") or ["tutorial_01"] * len(completions)
        rewards: list[float] = []
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
            records.append(
                {
                    "episode_id": uuid4().hex,
                    "timestamp": time.time(),
                    "level_id": lid,
                    "return": float(ret),
                    "won": bool(diag["won"]),
                    "steps": int(diag["steps"]),
                    "invalid": int(diag["invalid"]),
                    "n_actions_planned": int(diag["n_actions_planned"]),
                    "milestones": diag["milestones"],
                    "initial_prompt": prompt,
                    "completion": text,
                    "trajectory": diag["trajectory"],
                }
            )
        _write_records(log_path, records)
        if log_to_wandb:
            _log_records_to_wandb(records)
        return rewards

    return reward_func


def _write_records(path: Path | None, records: list[dict[str, Any]]) -> None:
    if path is None or not records:
        return
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _log_records_to_wandb(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    table = wandb.Table(
        columns=[
            "episode_id",
            "level_id",
            "return",
            "won",
            "steps",
            "invalid",
            "milestones",
            "initial_prompt",
            "completion",
            "trajectory_json",
        ]
    )
    for record in records:
        table.add_data(
            record["episode_id"],
            record["level_id"],
            record["return"],
            record["won"],
            record["steps"],
            record["invalid"],
            json.dumps(record["milestones"]),
            record["initial_prompt"],
            record["completion"],
            json.dumps(record["trajectory"], sort_keys=True),
        )
    wandb.log({"rollout/trajectories": table}, commit=False)
