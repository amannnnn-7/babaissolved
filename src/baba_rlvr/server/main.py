"""FastAPI app exposing the Baba Is You environment over the OpenEnv contract.

Endpoints:
    POST /reset        -> { session_id, observation }
    POST /step/{sid}   -> StepResponse
    GET  /state/{sid}  -> current observation
    POST /close/{sid}  -> { ok, episode_return, milestones }
    GET  /levels       -> list of registered level ids
"""

from __future__ import annotations

import os
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException

from .env import BabaEnv, list_levels
from .schemas import (
    BabaAction,
    CloseResponse,
    ResetRequest,
    ResetResponse,
    StepResponse,
)

app = FastAPI(
    title="Baba Is You — OpenEnv RLVR Server",
    version="0.1.0",
    description=(
        "OpenEnv-compatible HTTP environment for the puzzle game *Baba Is You*. "
        "Designed for GRPO/PPO training of LLM agents with verifiable rewards."
    ),
)

# Per-session env registry. Key = session_id (uuid4). Value = BabaEnv instance.
SESSIONS: dict[str, BabaEnv] = {}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "sessions": len(SESSIONS)}


@app.get("/levels")
def levels() -> list[str]:
    return list_levels()


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    level_id = req.level_id or "tutorial_01"
    if level_id not in list_levels():
        raise HTTPException(404, f"Unknown level_id: {level_id}")
    sid = str(uuid4())
    env = BabaEnv(level_id=level_id, use_memory=req.use_memory, seed=req.seed)
    obs = env.reset()
    SESSIONS[sid] = env
    return ResetResponse(session_id=sid, observation=obs)


@app.post("/step/{sid}", response_model=StepResponse)
def step(sid: str, action: BabaAction) -> StepResponse:
    if sid not in SESSIONS:
        raise HTTPException(404, "Unknown session_id")
    return SESSIONS[sid].step(action)


@app.get("/state/{sid}")
def state(sid: str):
    if sid not in SESSIONS:
        raise HTTPException(404, "Unknown session_id")
    env = SESSIONS[sid]
    return env._observe().model_dump()  # noqa: SLF001 — deliberate read-only debug


@app.post("/close/{sid}", response_model=CloseResponse)
def close(sid: str) -> CloseResponse:
    env = SESSIONS.pop(sid, None)
    if env is None:
        raise HTTPException(404, "Unknown session_id")
    ep_return, milestones = env.episode_summary()
    return CloseResponse(ok=True, episode_return=ep_return, milestones=milestones)


def run() -> None:
    """Console-script entry-point: `uv run baba-server`."""
    host = os.environ.get("BABA_HOST", "0.0.0.0")
    port = int(os.environ.get("BABA_PORT", "8000"))
    uvicorn.run("baba_rlvr.server.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
