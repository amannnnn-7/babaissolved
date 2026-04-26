#!/usr/bin/env bash
# Single-A100 short-demo run for the Baba Is You RLVR submission.
#
# What it does:
#   1) Build / verify the official 24-level pack (8 tiers x 3).
#   2) Start the OpenEnv FastAPI server.
#   3) GRPO-train Qwen2.5-1.5B-Instruct (LoRA, 4-bit, fp16) on the tier-ordered
#      training subset for ~150 steps, logging rich rollout metrics to W&B.
#   4) Run the base-vs-trained eval on the train + held-out (eval) levels.
#
# All knobs override-able via env vars; see defaults below.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://microsoft-research.wandb.io/}"
export WANDB_PROJECT="${WANDB_PROJECT:-baba-rlvr}"
export WANDB_NAME="${WANDB_NAME:-qwen2.5-baba-curriculum}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
ENV_URL="${BABA_ENV_URL:-http://127.0.0.1:8000}"
STEPS="${STEPS:-150}"
ADVANCE_WIN_RATE="${ADVANCE_WIN_RATE:-0.6}"
MIN_EPISODES_PER_TIER="${MIN_EPISODES_PER_TIER:-32}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-5e-6}"
OUT_DIR="${OUT_DIR:-ckpt-baba-curriculum}"
TRAJ_DIR="${TRAJ_DIR:-runs/qwen-curriculum-trajectories}"
EVAL_OUT="${EVAL_OUT:-runs/eval}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
SERVER_LOG="${SERVER_LOG:-/tmp/baba-server-curriculum.log}"

# Force a fresh W&B run id every invocation. ``WANDB_RESUME=never`` disables
# auto-resume; the unique id ensures no collision with prior runs.
export WANDB_RUN_ID="${WANDB_RUN_ID:-baba-$(date +%Y%m%d-%H%M%S)-$$}"
export WANDB_RESUME="${WANDB_RESUME:-never}"

# Wipe local wandb logs / trajectories / prior trajectory dir so this run
# starts from a clean slate. The remote portal history is untouched -- only
# local on-disk state is cleared.
if [[ "${KEEP_WANDB:-0}" != "1" ]]; then
  echo "[setup] clearing local wandb/ and previous trajectory dir"
  rm -rf wandb/ "$TRAJ_DIR" "$EVAL_OUT"
fi

# 1) Make sure pyBaba is available + build the official level pack.
if ! uv run python -c "import pyBaba" >/dev/null 2>&1; then
  echo "[setup] pyBaba not importable; running scripts/build_pybaba.sh"
  bash scripts/build_pybaba.sh
fi
uv run python scripts/build_official_levels.py

# 2) Boot the env server (background) and wait for /docs to come up.
uv run baba-server >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 60); do
  if curl -fsS "$ENV_URL/docs" >/dev/null; then break; fi
  sleep 1
done
curl -fsS "$ENV_URL/docs" >/dev/null

# 3) GRPO training — tier-ordered curriculum, fp16 to dodge unsloth dtype bug.
uv run python -m baba_rlvr.training.curriculum_train \
  --env-url "$ENV_URL" \
  --model "$MODEL_ID" \
  --out "$OUT_DIR" \
  --steps "$STEPS" \
  --advance-win-rate "$ADVANCE_WIN_RATE" \
  --min-episodes-per-tier "$MIN_EPISODES_PER_TIER" \
  --num-generations "$NUM_GENERATIONS" \
  --max-turns "$MAX_TURNS" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --learning-rate "$LR" \
  --trajectory-log-dir "$TRAJ_DIR" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_NAME"

# 4) Base vs trained evaluation on train + held-out levels.
# Use a distinct W&B run id so eval gets its own run, not a resume of training.
unset WANDB_RUN_ID
export WANDB_RUN_ID="baba-eval-$(date +%Y%m%d-%H%M%S)-$$"
uv run python -m baba_rlvr.eval.compare \
  --env-url "$ENV_URL" \
  --base "$MODEL_ID" \
  --trained "$OUT_DIR" \
  --episodes-per-level "$EVAL_EPISODES" \
  --max-turns "$MAX_TURNS" \
  --max-new-tokens "$MAX_COMPLETION_LENGTH" \
  --out "$EVAL_OUT" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "${WANDB_NAME}-eval"
