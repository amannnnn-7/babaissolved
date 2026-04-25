"""Pydantic schemas defining the OpenEnv contract for Baba Is You.

Two design rules drive these schemas:

1.  Observations carry **both** an LLM-friendly ASCII view *and* a strictly
    machine-readable token grid + active rule list. The LLM consumes the
    ASCII; the verifier consumes the structured fields. The agent cannot
    forge the verifier-visible state through clever generation.
2.  All actions are atomic single-step moves. Multi-step "macros" are not
    accepted server-side because GRPO needs reward signal at every step
    boundary for credit assignment.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    WAIT = "wait"


class BabaAction(BaseModel):
    action: ActionType
    rationale: str | None = Field(
        default=None,
        max_length=512,
        description="Optional CoT for analysis. NEVER consulted by the verifier.",
    )


class Rule(BaseModel):
    subject: str  # entity name e.g. "baba"
    verb: str = "IS"
    predicate: str  # property e.g. "YOU" or "WIN"


class BabaObservation(BaseModel):
    grid_ascii: str
    grid_tokens: list[list[str]]
    active_rules: list[Rule]
    you_entities: list[str]
    win_entities: list[str]
    step_count: int
    max_steps: int
    level_id: str
    # Optional agentic memory injected by the server (see memory/store.py).
    # Empty string when memory is disabled.
    memory_excerpt: str = ""


class StepResponse(BaseModel):
    observation: BabaObservation
    reward: float
    done: bool
    truncated: bool
    info: dict


class ResetRequest(BaseModel):
    level_id: str | None = None
    seed: int | None = None
    use_memory: bool = False


class ResetResponse(BaseModel):
    session_id: str
    observation: BabaObservation


class CloseResponse(BaseModel):
    ok: bool
    episode_return: float
    milestones: list[str]
