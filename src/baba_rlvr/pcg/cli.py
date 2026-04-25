"""CLI for procedural generation: `uv run baba-pcg generate ...`."""

from __future__ import annotations

import pickle
from pathlib import Path

import typer
from rich import print as rprint

from .map_elites import run_map_elites

app = typer.Typer(help="Baba Is You — procedural content generation.")


@app.command()
def generate(
    iterations: int = typer.Option(2000, help="MAP-Elites iterations."),
    out: Path = typer.Option(Path("levels/archive.pkl"), help="Output archive."),
    seed: int = typer.Option(0, help="RNG seed."),
    max_depth: int = typer.Option(25, help="Solver max depth (caps difficulty)."),
) -> None:
    """Generate a curriculum archive of solvable levels."""
    archive = run_map_elites(
        iterations=iterations, max_solver_depth=max_depth, rng_seed=seed
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(archive, f)
    rprint(
        f"[green]✓ wrote[/green] {out} — {len(archive)} elite cells, "
        f"max difficulty={max(e.difficulty for e in archive.values()):.1f}"
    )


@app.command()
def show(archive: Path = typer.Argument(..., help="Path to archive.pkl")) -> None:
    """Pretty-print the archive grid."""
    with archive.open("rb") as f:
        a = pickle.load(f)
    rprint(f"[bold]{len(a)} elite cells[/bold]")
    for cell, e in sorted(a.items()):
        rprint(
            f"  cell={cell}  level={e.level_id}  sol_len={e.sol_len}  "
            f"rules_mod={e.rules_modified}  diff={e.difficulty:.1f}"
        )


if __name__ == "__main__":
    app()
