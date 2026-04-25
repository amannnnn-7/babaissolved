"""Verifiable milestone-based reward tracker.

The tracker is the *verifier* in RLVR: it observes structured engine state
(active rules, you-entities, won flag) and emits a scalar reward per step.

Anti-hacking guarantees
-----------------------
1. Each named milestone fires **at most once per episode**.
2. Re-creating a rule that already appeared earlier in the episode yields
   nothing (`seen_rule_signatures`), preventing toggle-loop farming.
3. A small per-step cost makes any non-terminating loop strictly negative-EV.
4. The tracker reads only engine state — never LLM-produced text — so the
   agent cannot fabricate signal through prompt manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..server.schemas import BabaObservation

# Tunable in one place. Keep terminal reward >> any milestone sum so that
# winning is always the dominant gradient.
WEIGHT_WIN = 10.0
WEIGHT_DEATH = -2.0
WEIGHT_INVALID = -0.5
WEIGHT_STEP = -0.01

MILESTONE_WEIGHTS: dict[str, float] = {
    "first_rule_break": 1.0,
    "first_rule_make": 1.0,
    "self_redefine": 2.0,
    "win_condition_made": 1.5,
    "neutralized_kill": 1.0,
    "reached_win_tile": 0.0,  # subsumed by the WIN terminal; kept for logging
}


@dataclass
class RewardTracker:
    initial_rules: set[tuple[str, str]] = field(default_factory=set)
    triggered: set[str] = field(default_factory=set)
    seen_rule_signatures: set[tuple[str, str]] = field(default_factory=set)

    def init(self, obs: BabaObservation) -> None:
        self.initial_rules = {(r.subject, r.predicate) for r in obs.active_rules}
        self.seen_rule_signatures = set(self.initial_rules)
        self.triggered.clear()

    def compute(
        self,
        prev: BabaObservation,
        curr: BabaObservation,
        *,
        done: bool,
        won: bool,
        invalid: bool,
        died: bool,
    ) -> tuple[float, dict]:
        r = 0.0
        info: dict = {"milestones": []}

        # Always-on per-step shaping (small, prevents loops).
        r += WEIGHT_STEP
        if invalid:
            r += WEIGHT_INVALID
        if died:
            r += WEIGHT_DEATH

        prev_set = {(x.subject, x.predicate) for x in prev.active_rules}
        curr_set = {(x.subject, x.predicate) for x in curr.active_rules}
        broken = prev_set - curr_set
        made = curr_set - prev_set

        def fire(name: str) -> float:
            if name in self.triggered:
                return 0.0
            self.triggered.add(name)
            val = MILESTONE_WEIGHTS[name]
            info["milestones"].append((name, val))
            return val

        # Broke an originally-active rule.
        if broken & self.initial_rules:
            r += fire("first_rule_break")

        # Made a rule novel to the whole episode.
        novel_made = made - self.seen_rule_signatures
        self.seen_rule_signatures |= made
        if novel_made:
            r += fire("first_rule_make")
            if any(p == "WIN" for _, p in novel_made):
                r += fire("win_condition_made")

        # Identity rewrite: which entity is YOU changed.
        if set(prev.you_entities) != set(curr.you_entities):
            r += fire("self_redefine")

        prev_kills = {s for s, p in prev_set if p in ("KILL", "DEFEAT")}
        curr_kills = {s for s, p in curr_set if p in ("KILL", "DEFEAT")}
        if prev_kills - curr_kills:
            r += fire("neutralized_kill")

        if won:
            r += WEIGHT_WIN
            info["milestones"].append(("WIN", WEIGHT_WIN))

        info["reward"] = r
        info["done"] = done
        return r, info
