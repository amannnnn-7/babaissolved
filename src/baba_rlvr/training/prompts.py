"""Prompt-building helpers shared between training and eval."""

from __future__ import annotations

import json
import re

from ..server.schemas import ActionType, BabaObservation

SYSTEM_PROMPT = (
    "You are an agent playing *Baba Is You*. The world is a grid; rules are formed "
    "by SUBJECT IS PROPERTY text triples. Pushing text blocks REWRITES the rules.\n"
    "Goal: become an entity that overlaps a WIN entity.\n"
    "Output EXACTLY one JSON object on a single line, e.g.\n"
    '  {"action": "up", "rationale": "push BABA out of YOU"}\n'
    'Allowed actions: "up", "down", "left", "right", "wait". '
    "Keep rationale under 30 words."
)


def build_prompt(obs: BabaObservation) -> str:
    rules = ", ".join(f"{r.subject} IS {r.predicate}" for r in obs.active_rules) or "(none)"
    mem = ""
    if obs.memory_excerpt:
        mem = f"\n# Verified lessons from previous episodes\n{obs.memory_excerpt}\n"
    return (
        f"{SYSTEM_PROMPT}\n{mem}\n"
        f"Step {obs.step_count}/{obs.max_steps}\n"
        f"You control: {obs.you_entities}\n"
        f"Win entities: {obs.win_entities}\n"
        f"Active rules: {rules}\n"
        f"Grid:\n{obs.grid_ascii}\n\n"
        f"Your move:"
    )


_ACTION_RE = re.compile(r'"action"\s*:\s*"(up|down|left|right|wait)"', re.IGNORECASE)


def parse_action(text: str) -> tuple[ActionType, bool]:
    """Parse an LLM completion into an ActionType.

    Returns (action, was_well_formed). On failure we default to WAIT and the
    server applies the invalid-action penalty via the reward tracker.
    """
    text = text.strip()
    # Try strict JSON first.
    try:
        first_line = text.splitlines()[0]
        obj = json.loads(first_line)
        if isinstance(obj, dict) and "action" in obj:
            return ActionType(obj["action"].lower()), True
    except (json.JSONDecodeError, ValueError, KeyError, IndexError):
        pass
    # Fall back to regex.
    m = _ACTION_RE.search(text)
    if m:
        try:
            return ActionType(m.group(1).lower()), True
        except ValueError:
            pass
    return ActionType.WAIT, False
