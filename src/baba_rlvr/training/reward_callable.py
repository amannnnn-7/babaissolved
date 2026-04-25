"""GRPO reward callable: spins a full multi-turn episode against the env server.

Wired so that TRL's GRPOTrainer treats each prompt as the *initial* state of
an episode. The completion contributes the first action; the reward callable
then drives the rest of the episode using its own quick-decode of the model
provided via a generation closure. The trajectory return becomes the scalar
GRPO reward — the group baseline normalizes it across the K rollouts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from ..server.schemas import ActionType, BabaAction, BabaObservation
from .prompts import build_prompt, parse_action

GenerateFn = Callable[[str], str]


def episode_return(
    env_url: str,
    level_id: str,
    first_completion: str,
    generate: GenerateFn,
    max_turns: int = 40,
    gamma: float = 0.99,
    use_memory: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Roll out one episode; return discounted return + diagnostics."""
    diag: dict[str, Any] = {"won": False, "steps": 0, "milestones": [], "invalid": 0}
    with httpx.Client(timeout=60, base_url=env_url) as cx:
        r = cx.post("/reset", json={"level_id": level_id, "use_memory": use_memory}).json()
        sid = r["session_id"]
        completion = first_completion
        ret = 0.0
        for t in range(max_turns):
            action_kind, well_formed = parse_action(completion)
            if not well_formed:
                diag["invalid"] += 1
            resp = cx.post(
                f"/step/{sid}",
                json=BabaAction(action=action_kind).model_dump(),
            ).json()
            ret += (gamma ** t) * resp["reward"]
            for m, _ in resp["info"].get("milestones", []):
                if m not in diag["milestones"]:
                    diag["milestones"].append(m)
            diag["steps"] = t + 1
            if resp["done"]:
                diag["won"] = resp["info"].get("won", False)
                break
            obs = BabaObservation(**resp["observation"])
            completion = generate(build_prompt(obs))
        cx.post(f"/close/{sid}")
    return ret, diag


def make_reward_func(
    env_url: str,
    generate: GenerateFn,
    *,
    max_turns: int = 40,
    use_memory: bool = False,
) -> Callable[..., list[float]]:
    """Factory that returns a TRL-compatible reward function.

    TRL's GRPOTrainer calls reward_funcs(prompts, completions, **kwargs) and
    expects a list[float] of length len(completions). We additionally expect
    a `level_id` column from the dataset, surfaced in **kwargs.
    """

    def reward_func(prompts, completions, **kwargs) -> list[float]:
        level_ids = kwargs.get("level_id") or ["tutorial_01"] * len(completions)
        rewards: list[float] = []
        for comp, lid in zip(completions, level_ids, strict=False):
            text = comp if isinstance(comp, str) else comp[0]["content"]
            ret, _diag = episode_return(
                env_url=env_url,
                level_id=lid,
                first_completion=text,
                generate=generate,
                max_turns=max_turns,
                use_memory=use_memory,
            )
            rewards.append(float(ret))
        return rewards

    return reward_func
