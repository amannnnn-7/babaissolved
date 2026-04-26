"""Prompt-building helpers shared between training and eval.

We use a *plan-mode* protocol: the model first thinks (free-form) and then
emits a sequence of moves after a ``PLAN:`` marker, one per line. The reward
function parses the plan and steps the env action-by-action. The trajectory
return is the scalar reward for GRPO, so every action token in the completion
shares the same advantage -- this gives credit assignment to the whole plan,
not just the first action.
"""

from __future__ import annotations

import json
import re

from ..server.schemas import ActionType, BabaObservation

# Hard cap on plan length the prompt advertises. The reward function further
# clamps to ``max_turns`` to bound rollout cost.
MAX_PLAN_LEN = 30

SYSTEM_PROMPT = (
    "You are an agent playing *Baba Is You*. The world is a grid; rules are "
    "formed by SUBJECT IS PROPERTY text triples. Pushing a text block REWRITES "
    "the rules. Text blocks are always pushable; physical objects are pushable "
    "only when an active rule says so.\n"
    "Early mechanics include AND rules, STOP walls, PUSH rocks, SINK water, "
    "OPEN/SHUT key+door pairs, and changing which entity IS YOU.\n"
    "Goal: become an entity that overlaps a WIN entity.\n\n"
    "Think step-by-step about which rules to break, form, or exploit, then "
    f"output a sequence of at most {MAX_PLAN_LEN} moves.\n"
    "Format your answer EXACTLY as:\n"
    "REASONING: <your concise plan in plain text, one or two sentences>\n"
    "PLAN:\n"
    "<move>\n<move>\n...\n"
    "Each <move> is one of: up, down, left, right, wait (lowercase, one per line). "
    "After PLAN: output ONLY moves -- no numbering, no commentary.\n"
    "Example:\n"
    "REASONING: rock blocks the flag, push it left then walk down to WIN.\n"
    "PLAN:\nleft\nleft\ndown\ndown\n"
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
        f"Your answer:"
    )


_VALID_ACTION_TOKENS = {a.value for a in ActionType}
_ACTION_WORD_RE = re.compile(r"\b(up|down|left|right|wait)\b", re.IGNORECASE)
_PLAN_MARKER_RE = re.compile(r"PLAN\s*:\s*", re.IGNORECASE)
_STRIP_CHARS = " \t\"',.;:!?[]{}()-*#`"


def parse_plan(text: str, max_actions: int = MAX_PLAN_LEN) -> tuple[list[ActionType], int]:
    """Parse a plan-mode completion into a list of actions.

    If a ``PLAN:`` marker is present, only the text after it is scanned.
    Otherwise we scan the whole completion (forgiving fallback).

    Returns (actions, invalid_line_count). A line counts as invalid if it is
    non-empty after stripping whitespace/punctuation but contains no
    recognizable action word.
    """
    m = _PLAN_MARKER_RE.search(text)
    body = text[m.end():] if m else text

    actions: list[ActionType] = []
    invalid = 0
    for raw in body.splitlines():
        if len(actions) >= max_actions:
            break
        line = raw.strip().strip(_STRIP_CHARS).lower()
        if not line:
            continue
        if line in _VALID_ACTION_TOKENS:
            actions.append(ActionType(line))
            continue
        word = _ACTION_WORD_RE.search(line)
        if word:
            actions.append(ActionType(word.group(1).lower()))
        else:
            invalid += 1
    return actions, invalid


_ACTION_RE = re.compile(r'"action"\s*:\s*"(up|down|left|right|wait)"', re.IGNORECASE)


def parse_action(text: str) -> tuple[ActionType, bool]:
    """Parse a single action from a completion (legacy single-action protocol).

    Kept for backward compatibility with older entry points (e.g. the play UI).
    Returns (action, was_well_formed). On failure defaults to WAIT.
    """
    text = text.strip()
    try:
        first_line = text.splitlines()[0]
        obj = json.loads(first_line)
        if isinstance(obj, dict) and "action" in obj:
            return ActionType(obj["action"].lower()), True
    except (json.JSONDecodeError, ValueError, KeyError, IndexError):
        pass
    m = _ACTION_RE.search(text)
    if m:
        try:
            return ActionType(m.group(1).lower()), True
        except ValueError:
            pass
    actions, _ = parse_plan(text, max_actions=1)
    if actions:
        return actions[0], True
    return ActionType.WAIT, False
