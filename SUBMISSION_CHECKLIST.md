# Submission Checklist

This file is the last-pass checklist for turning the repository into a valid
OpenEnv Hackathon submission without dragging training artifacts into the
Space.

## Current Repository State

- [x] OpenEnv-style FastAPI environment exists
- [x] Hugging Face Docker Space scaffold exists (`openenv.yaml`, `Dockerfile`, `server.py`)
- [x] Colab-friendly training notebook exists: `notebooks/colab_grpo_demo.ipynb`
- [x] Main single-A100 training launcher exists: `scripts/run_qwen_a100_grpo.sh`
- [x] Root README is structured as a submission landing page
- [x] Detailed blog draft exists: `blog_draft.md`
- [ ] Live Hugging Face Space URL added to `README.md`
- [ ] Public W&B run link or committed reward/loss plots added to `README.md`
- [ ] Mini-blog or short video URL added to `README.md`
- [ ] Final numeric results copied into `README.md` and `blog_draft.md`

## Before Pushing The Space

1. Use a dedicated Hugging Face Space repo, not the training workspace remote.
2. Keep local artifacts out of the Space push: `ckpt-*`, `runs/`, `wandb/`, generated caches, and notebooks outputs.
3. Confirm the Space will build from the tracked code only; `.dockerignore` already excludes the heavy local context.
4. Make sure the repo you push contains the root `README.md`, `openenv.yaml`, `Dockerfile`, `server.py`, `src/`, `levels/`, and `scripts/patches/`.

## Publish Flow

1. Create a Docker Space on Hugging Face.
2. Push the repo contents to that Space.
3. Verify these live routes:
   - `/health`
   - `/docs`
   - `/play`
4. Paste the live Space URL into the README submission table.
5. If you use the OpenEnv CLI, the manifest is ready for `openenv push --repo-id <user-or-org>/baba-rlvr`.

## Evidence To Add After The Current Run Finishes

1. Reward curve with labeled x and y axes.
2. Loss curve with labeled x and y axes.
3. Base-vs-trained comparison on held-out levels.
4. Public W&B run link or checked-in PNGs.
5. One short paragraph in the README explaining what improved.

## README Final Pass

1. Replace every `TODO before submission` placeholder with a real public URL.
2. Add at least one public training-evidence link.
3. Add the Hugging Face post or YouTube link.
4. Add a short result summary with concrete numbers.
5. Re-read the first screen of the README and ensure a judge can understand the problem, the environment, and the result in under 3 minutes.

## Safety Check

Do not interrupt or modify the ongoing run launched from `scripts/run_qwen_a100_grpo.sh`. All submission work should stay in the packaging, documentation, and deployment layer until that run has finished.