"""Per-session environment wrapper.

Wraps the pure-Python `engine.World` with reward bookkeeping and OpenEnv-
shaped observation conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engine import Direction, World
from ..levels.loader import LEVEL_REGISTRY, load_level
from ..memory.store import MemoryStore
from ..reward.tracker import RewardTracker
from .schemas import BabaAction, BabaObservation, Rule, StepResponse


@dataclass
class BabaEnv:
    level_id: str
    use_memory: bool = False
    seed: int | None = None
    world: World = field(init=False)
    tracker: RewardTracker = field(default_factory=RewardTracker)
    memory: MemoryStore | None = field(default=None)
    _prev_obs: BabaObservation | None = field(default=None, init=False)
    _episode_return: float = 0.0
    _triggered: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.world = load_level(self.level_id)
        if self.use_memory:
            self.memory = MemoryStore(level_id=self.level_id)

    # ----------------------------------------------------- public API
    def reset(self) -> BabaObservation:
        self.world = load_level(self.level_id)
        self.tracker = RewardTracker()
        self._episode_return = 0.0
        self._triggered = []
        obs = self._observe()
        self.tracker.init(obs)
        self._prev_obs = obs
        return obs

    def step(self, action: BabaAction) -> StepResponse:
        info = self.world.step(Direction(action.action.value))
        curr = self._observe()
        prev = self._prev_obs or curr
        reward, rinfo = self.tracker.compute(
            prev=prev,
            curr=curr,
            done=self.world.won or self.world.lost,
            won=self.world.won,
            invalid=info.get("invalid_move", False),
            died=info.get("died", False),
        )
        self._episode_return += reward
        for name, _ in rinfo.get("milestones", []):
            if name not in self._triggered:
                self._triggered.append(name)
                if self.memory:
                    self.memory.note_milestone(name, curr)

        truncated = bool(info.get("truncated", False))
        done = self.world.won or self.world.lost or truncated

        self._prev_obs = curr
        return StepResponse(
            observation=curr,
            reward=reward,
            done=done,
            truncated=truncated,
            info=rinfo | {"won": self.world.won, "lost": self.world.lost},
        )

    def episode_summary(self) -> tuple[float, list[str]]:
        if self.memory:
            self.memory.flush(won=self.world.won)
        return self._episode_return, list(self._triggered)

    # ----------------------------------------------------- helpers
    def _observe(self) -> BabaObservation:
        rules = [
            Rule(subject=e.value, predicate=p.value) for e, p in sorted(self.world.rules)
        ]
        memory_excerpt = self.memory.read() if self.memory else ""
        return BabaObservation(
            grid_ascii=self.world.render_ascii(),
            grid_tokens=self.world.to_tokens(),
            active_rules=rules,
            you_entities=sorted(e.value for e in self.world.you_entities()),
            win_entities=sorted(e.value for e in self.world.win_entities()),
            step_count=self.world.step_count,
            max_steps=self.world.max_steps,
            level_id=self.level_id,
            memory_excerpt=memory_excerpt,
        )


def list_levels() -> list[str]:
    return sorted(LEVEL_REGISTRY.keys())
