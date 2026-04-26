"""CLI for procedural generation: `uv run baba-pcg generate ...`."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from .diverse import generate_diverse_levels
from .map_elites import run_map_elites

app = typer.Typer(help="Baba Is You — procedural content generation.")
_DEFAULT_ARCHIVE = Path("levels/archive.pkl")
_DEFAULT_GENERATED_DIR = Path("levels/_generated")


@app.command()
def generate(
    iterations: Annotated[int, typer.Option(help="MAP-Elites iterations.")] = 2000,
    out: Annotated[Path, typer.Option(help="Output archive.")] = _DEFAULT_ARCHIVE,
    seed: Annotated[int, typer.Option(help="RNG seed.")] = 0,
    max_depth: Annotated[int, typer.Option(help="Solver max depth (caps difficulty).")] = 25,
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
def diverse(
    count: Annotated[int, typer.Option(help="Number of diverse generated levels.")] = 50,
    out_dir: Annotated[
        Path, typer.Option(help="Directory for .txt maps.")
    ] = _DEFAULT_GENERATED_DIR,
    prefix: Annotated[str, typer.Option(help="Filename/level-id prefix.")] = "diverse",
    max_depth: Annotated[int, typer.Option(help="Solver max depth.")] = 50,
) -> None:
    """Generate a solver-checked, visibly diverse level set."""
    levels = generate_diverse_levels(
        count=count,
        out_dir=out_dir,
        prefix=prefix,
        max_solver_depth=max_depth,
    )
    families = sorted({level.family for level in levels})
    rprint(
        f"[green]✓ wrote[/green] {len(levels)} levels to {out_dir} "
        f"across {len(families)} families: {', '.join(families)}"
    )


@app.command()
def show(archive: Annotated[Path, typer.Argument(help="Path to archive.pkl")]) -> None:
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
