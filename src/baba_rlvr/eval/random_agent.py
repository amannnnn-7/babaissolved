"""Standalone random-agent runner that talks to the engine in-process.

Used when you want to sanity-check rewards without standing up the server.
"""

from __future__ import annotations

import argparse
import random as _random

from ..engine import Direction
from ..levels.loader import LEVEL_REGISTRY
from ..server.env import BabaEnv
from ..server.schemas import ActionType, BabaAction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--level-id", default="tutorial_01")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.level_id not in LEVEL_REGISTRY:
        raise SystemExit(f"Unknown level: {args.level_id}. Have: {sorted(LEVEL_REGISTRY)}")

    rng = _random.Random(args.seed)
    actions = [a for a in ActionType if a != ActionType.WAIT]
    wins = 0
    for ep in range(args.episodes):
        env = BabaEnv(level_id=args.level_id)
        env.reset()
        ep_ret = 0.0
        for _ in range(80):
            resp = env.step(BabaAction(action=rng.choice(actions)))
            ep_ret += resp.reward
            if resp.done:
                break
        won = env.world.won
        wins += int(won)
        print(f"ep {ep:3d}  return={ep_ret:+.2f}  won={won}  steps={env.world.step_count}")
    print(f"\nSuccess: {wins}/{args.episodes} = {wins / args.episodes:.1%}")


if __name__ == "__main__":
    main()
