from baba_rlvr.reward.tracker import RewardTracker
from baba_rlvr.server.schemas import BabaObservation, Rule


def _obs(rules, you=("baba",)):
    return BabaObservation(
        grid_ascii="",
        grid_tokens=[],
        active_rules=[Rule(subject=s, predicate=p) for s, p in rules],
        you_entities=list(you),
        win_entities=["flag"],
        step_count=0,
        max_steps=80,
        level_id="t",
    )


def test_step_cost_only():
    t = RewardTracker()
    obs = _obs([("baba", "YOU"), ("flag", "WIN")])
    t.init(obs)
    r, info = t.compute(obs, obs, done=False, won=False, invalid=False, died=False)
    assert r < 0  # step cost
    assert info["milestones"] == []


def test_first_rule_break_fires_once():
    t = RewardTracker()
    prev = _obs([("baba", "YOU"), ("wall", "STOP")])
    t.init(prev)
    curr = _obs([("baba", "YOU")])  # broke wall-stop
    r1, info1 = t.compute(prev, curr, done=False, won=False, invalid=False, died=False)
    assert any(m == "first_rule_break" for m, _ in info1["milestones"])
    # Re-introduce and re-break: should not fire again.
    r2, info2 = t.compute(prev, curr, done=False, won=False, invalid=False, died=False)
    assert all(m != "first_rule_break" for m, _ in info2["milestones"])
    assert r2 < r1  # only step cost remains, no milestone bonus


def test_win_dominates():
    t = RewardTracker()
    prev = _obs([("baba", "YOU"), ("flag", "WIN")])
    t.init(prev)
    r, info = t.compute(prev, prev, done=True, won=True, invalid=False, died=False)
    assert r >= 9.0  # WIN weight is 10 minus the small step cost
    assert any(m == "WIN" for m, _ in info["milestones"])


def test_loop_farming_blocked():
    """Repeatedly toggling the same rule yields no extra reward."""
    t = RewardTracker()
    initial = _obs([("baba", "YOU")])
    t.init(initial)
    with_extra = _obs([("baba", "YOU"), ("rock", "PUSH")])
    # First creation: novel.
    r1, _ = t.compute(initial, with_extra, done=False, won=False, invalid=False, died=False)
    # Break it back.
    r2, _ = t.compute(with_extra, initial, done=False, won=False, invalid=False, died=False)
    # Re-create the same rule: now "seen", so no first_rule_make bonus.
    r3, info3 = t.compute(initial, with_extra, done=False, won=False, invalid=False, died=False)
    assert r1 > r3
    assert all(m != "first_rule_make" for m, _ in info3["milestones"])
