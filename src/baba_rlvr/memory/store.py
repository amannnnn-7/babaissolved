"""Agentic memory scratchpad (lessons.md).

Hybrid design: RL gradients update the LLM weights *and* the agent
maintains a small markdown scratchpad of milestone-verified lessons across
episodes. Crucially, the LLM cannot write arbitrary text into memory — only
the verifier appends entries when a real milestone fires. This keeps the
memory grounded in actual environment events.

Toggle via reset(use_memory=True) on the OpenEnv server. Used for the
ablation study (with-memory vs without-memory success rate).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from ..server.schemas import BabaObservation

# Lines of memory injected into the prompt every step. Capped to keep the
# context window small.
MAX_MEMORY_LINES = 20

DEFAULT_DIR = Path(os.environ.get("BABA_MEMORY_DIR", "memory_runs"))


class MemoryStore:
    def __init__(self, level_id: str, root: Path | None = None) -> None:
        self.level_id = level_id
        self.root = root or DEFAULT_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "lessons.md"
        self._pending: list[str] = []
        if not self.path.exists():
            self.path.write_text("# Baba Is You — Verified Lessons\n\n")

    # ---------------------------------------------------- read
    def read(self) -> str:
        """Return the last MAX_MEMORY_LINES lines of the scratchpad."""
        if not self.path.exists():
            return ""
        lines = self.path.read_text().splitlines()
        return "\n".join(lines[-MAX_MEMORY_LINES:])

    # ---------------------------------------------------- write
    def note_milestone(self, name: str, obs: BabaObservation) -> None:
        rules = ", ".join(f"{r.subject} IS {r.predicate}" for r in obs.active_rules)
        self._pending.append(
            f"- [{name}] level={obs.level_id} step={obs.step_count} "
            f"you={obs.you_entities} rules=({rules})"
        )

    def flush(self, *, won: bool) -> None:
        if not self._pending:
            return
        ts = datetime.utcnow().isoformat(timespec="seconds")
        outcome = "WIN" if won else "loss"
        with self.path.open("a") as f:
            f.write(f"\n## {ts} — {self.level_id} — {outcome}\n")
            for line in self._pending:
                f.write(line + "\n")
        self._pending.clear()
