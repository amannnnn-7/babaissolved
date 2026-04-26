"""FastAPI app exposing the Baba Is You environment over the OpenEnv contract.

Endpoints:
    POST /reset        -> { session_id, observation }
    POST /step/{sid}   -> StepResponse
    GET  /state/{sid}  -> current observation
    POST /close/{sid}  -> { ok, episode_return, milestones }
    GET  /levels       -> list of registered level ids

Browser play mode (human eval / demo):
    GET  /play              -> interactive HTML (arrow keys)
    GET  /play/frame/{sid}.png -> PNG of current state
    GET  /play/solve/{lvl}  -> { actions: [...] } from BFS solver
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from ..engine import Direction  # noqa: F401  (re-exported for convenience)
from ..levels.loader import load_level
from ..pcg.solver import bfs_solve
from ..viz.renderer import render_world
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

STATIC_DIR = Path(__file__).parent / "static"


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


# ---------------------------------------------------------------------------
# Browser play mode — single-page HTML with arrow-key controls.
# ---------------------------------------------------------------------------
@app.get("/")
def index_redirect() -> Response:
    return Response(
        status_code=302,
        headers={"location": "/play"},
    )


@app.get("/play")
def play_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "play.html", media_type="text/html")


@app.get("/play/frame/{sid}.png")
def play_frame(sid: str) -> Response:
    if sid not in SESSIONS:
        raise HTTPException(404, "Unknown session_id")
    env = SESSIONS[sid]
    img = render_world(
        env.world,
        cell=56,
        title=env.level_id,
        step_idx=env.world.step_count,
        backend="sprites",
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/play/solve/{level_id}")
def play_solve(level_id: str, max_depth: int = 25) -> dict:
    if level_id not in list_levels():
        raise HTTPException(404, f"Unknown level_id: {level_id}")
    world = load_level(level_id)
    sol = bfs_solve(world, max_depth=max_depth)
    if sol is None:
        return {"actions": None, "reason": f"no solution within depth {max_depth}"}
    return {"actions": [a.value for a in sol]}


def run() -> None:
    """Console-script entry-point: `uv run baba-server`."""
    host = os.environ.get("BABA_HOST", "0.0.0.0")
    port = int(os.environ.get("BABA_PORT", "8000"))
    uvicorn.run("baba_rlvr.server.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
