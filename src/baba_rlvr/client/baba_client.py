"""Thin client for the OpenEnv Baba server.

Useful for notebooks and ad-hoc scripts. Keeps a session_id and exposes a
`gym`-style step()/reset() API.
"""

from __future__ import annotations

import httpx

from ..server.schemas import ActionType, BabaAction, BabaObservation, StepResponse


class BabaClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30) -> None:
        self.cx = httpx.Client(base_url=base_url, timeout=timeout)
        self.session_id: str | None = None

    def reset(self, level_id: str = "tutorial_01", use_memory: bool = False) -> BabaObservation:
        r = self.cx.post(
            "/reset", json={"level_id": level_id, "use_memory": use_memory}
        ).json()
        self.session_id = r["session_id"]
        return BabaObservation(**r["observation"])

    def step(self, action: ActionType | str) -> StepResponse:
        if self.session_id is None:
            raise RuntimeError("Call reset() first.")
        if isinstance(action, str):
            action = ActionType(action)
        r = self.cx.post(
            f"/step/{self.session_id}",
            json=BabaAction(action=action).model_dump(),
        ).json()
        return StepResponse(**r)

    def close(self) -> dict:
        if self.session_id is None:
            return {"ok": True}
        r = self.cx.post(f"/close/{self.session_id}").json()
        self.session_id = None
        return r

    def __enter__(self) -> BabaClient:
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.close()
        finally:
            self.cx.close()
