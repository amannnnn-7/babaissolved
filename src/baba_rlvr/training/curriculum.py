"""Tier-based curriculum for the official Baba Is You level pack.

The official pack lives under ``levels/official/`` and is named so we can
group it into 8 difficulty tiers (T1..T8) of 3 levels each. Within each tier
the level whose id ends in ``_eval`` is held out: it is **never** trained on
and is reserved for the base-vs-trained evaluation harness.

This module is the single source of truth for which levels are training and
which are eval; both ``training/curriculum_train.py`` and
``eval/cli.py:bench`` import from here.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tiers — keep aligned with scripts/build_official_levels.py
# ---------------------------------------------------------------------------

TIER_LABELS: dict[str, str] = {
    "T1": "Movement",
    "T2": "STOP walls",
    "T3": "PUSH rocks",
    "T4": "Break a STOP rule",
    "T5": "Water / SINK",
    "T6": "Form a WIN rule",
    "T7": "Schema drift (IS YOU)",
    "T8": "OPEN / SHUT (key/door)",
}

TIERS: dict[str, list[str]] = {
    "T1": ["t1_01_baba_is_you", "t1_02_step_up", "t1_03_corner_eval"],
    "T2": ["t2_01_wall_gap", "t2_02_around_wall", "t2_03_corridor_eval"],
    "T3": ["t3_01_push_rock", "t3_02_push_two", "t3_03_push_eval"],
    "T4": ["t4_01_break_stop", "t4_02_push_stop_aside", "t4_03_break_stop_eval"],
    "T5": ["t5_01_water_detour", "t5_02_sink_with_rock", "t5_03_water_eval"],
    "T6": ["t6_01_make_flag_win", "t6_02_make_rock_win", "t6_03_make_win_eval"],
    "T7": ["t7_01_become_rock", "t7_02_swap_you", "t7_03_drift_eval"],
    "T8": ["t8_01_key_door", "t8_02_grab_key_first", "t8_03_door_eval"],
}


@dataclass(frozen=True)
class CurriculumSplit:
    train: list[str]
    eval: list[str]
    tier_of: dict[str, str]


def split() -> CurriculumSplit:
    train: list[str] = []
    eval_: list[str] = []
    tier_of: dict[str, str] = {}
    for tier, lvls in TIERS.items():
        for lid in lvls:
            tier_of[lid] = tier
            (eval_ if lid.endswith("_eval") else train).append(lid)
    return CurriculumSplit(train=train, eval=eval_, tier_of=tier_of)


def tier_order() -> list[str]:
    return list(TIERS.keys())
