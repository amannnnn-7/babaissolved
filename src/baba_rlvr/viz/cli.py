"""CLI: `uv run baba-viz play <level> --actions urrrd --out demo.gif`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..engine import Direction
from ..levels.loader import LEVEL_REGISTRY, load_level
from ..pcg.solver import bfs_solve
from .renderer import (
    actions_from_strings,
    render_trajectory_gif,
    render_trajectory_strip,
    render_world,
    rollout_actions,
)

app = typer.Typer(help="Baba Is You — trajectory & gameplay visualizer.")


@app.command()
def frame(
    level_id: str = typer.Argument(..., help="Registered level id."),
    out: Path = typer.Option(Path("frame.png")),
    cell: int = typer.Option(48),
) -> None:
    """Render the initial state of a level to a PNG."""
    if level_id not in LEVEL_REGISTRY:
        raise typer.BadParameter(f"Unknown level: {level_id}")
    world = load_level(level_id)
    img = render_world(world, cell=cell, title=level_id, step_idx=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    rprint(f"[green]✓[/green] wrote {out}")


@app.command()
def play(
    level_id: str = typer.Argument(...),
    actions: str = typer.Option(
        "",
        help="Action string. Single-char shortcuts: u d l r w. e.g. 'rrrrrr'",
    ),
    out: Path = typer.Option(Path("trajectory.gif")),
    cell: int = typer.Option(48),
    duration_ms: int = typer.Option(400),
) -> None:
    """Render an animated GIF of a trajectory through a level."""
    world = load_level(level_id)
    dirs = actions_from_strings(list(actions)) if actions else []
    if not dirs:
        raise typer.BadParameter("Provide --actions, e.g. --actions rrrrrr")
    frames = rollout_actions(world, dirs)
    render_trajectory_gif(frames, out, cell=cell, title=level_id, duration_ms=duration_ms)
    rprint(f"[green]✓[/green] wrote {out} ({len(frames)} frames)")


@app.command()
def solve(
    level_id: str = typer.Argument(...),
    out: Path = typer.Option(Path("solution.gif")),
    max_depth: int = typer.Option(20),
    cell: int = typer.Option(48),
) -> None:
    """BFS-solve the level and animate the optimal solution."""
    world = load_level(level_id)
    sol = bfs_solve(world, max_depth=max_depth)
    if sol is None:
        rprint(f"[red]✗ no solution found within depth {max_depth}[/red]")
        raise typer.Exit(1)
    frames = rollout_actions(world, sol)
    render_trajectory_gif(frames, out, cell=cell, title=f"{level_id} — solver", duration_ms=350)
    rprint(
        f"[green]✓[/green] solved in {len(sol)} moves: "
        f"{''.join(a.value[0] for a in sol)} -> {out}"
    )


@app.command()
def strip(
    level_id: str = typer.Argument(...),
    actions: str = typer.Option("", help="Action string (u/d/l/r/w)."),
    out: Path = typer.Option(Path("strip.png")),
    cols: int = typer.Option(6),
    cell: int = typer.Option(32),
) -> None:
    """Render all trajectory frames as a single PNG strip (good for blogs)."""
    world = load_level(level_id)
    if actions:
        dirs = actions_from_strings(list(actions))
    else:
        dirs = bfs_solve(world, max_depth=20) or []
        if not dirs:
            raise typer.BadParameter("No --actions and BFS solver failed.")
    frames = rollout_actions(world, dirs)
    render_trajectory_strip(frames, out, cell=cell, cols=cols, title=level_id)
    rprint(f"[green]✓[/green] wrote {out}")


if __name__ == "__main__":
    app()
