"""CLI: `uv run baba-eval random` etc.

Used to (a) sanity-check the env server, and (b) compute baseline numbers
(random-agent success rate) for the ablation table in the pitch deck.
"""

from __future__ import annotations

import random as _random

import httpx
import typer
from rich import print as rprint

from ..server.schemas import ActionType

app = typer.Typer(help="Baba Is You — evaluation harness.")

ACTIONS = [a.value for a in ActionType if a != ActionType.WAIT]


@app.command()
def random(
    url: str = typer.Option("http://localhost:8000", help="OpenEnv base URL."),
    level_id: str = typer.Option("tutorial_01"),
    episodes: int = typer.Option(20),
    seed: int = typer.Option(0),
) -> None:
    """Run a random agent and report success rate."""
    rng = _random.Random(seed)
    wins = 0
    returns = []
    with httpx.Client(timeout=30, base_url=url) as cx:
        for ep in range(episodes):
            r = cx.post("/reset", json={"level_id": level_id}).json()
            sid = r["session_id"]
            obs = r["observation"]
            done = False
            ep_ret = 0.0
            while not done:
                a = {"action": rng.choice(ACTIONS)}
                resp = cx.post(f"/step/{sid}", json=a).json()
                ep_ret += resp["reward"]
                done = resp["done"]
                obs = resp["observation"]
                if obs["step_count"] >= obs["max_steps"]:
                    done = True
            close = cx.post(f"/close/{sid}").json()
            won = "WIN" in close["milestones"]
            wins += int(won)
            returns.append(ep_ret)
            rprint(f"ep {ep:3d}  return={ep_ret:+.2f}  won={won}")
    rprint(
        f"\n[bold]Success rate:[/bold] {wins}/{episodes} = "
        f"{wins / episodes:.1%}  mean return={sum(returns) / len(returns):+.2f}"
    )


@app.command()
def levels(url: str = typer.Option("http://localhost:8000")) -> None:
    """List levels registered on the server."""
    r = httpx.get(f"{url}/levels").json()
    for lvl in r:
        rprint(f"  - {lvl}")


if __name__ == "__main__":
    app()
