#!/usr/bin/env bash
# Single-node 8-GPU GRPO launcher.
#
# Required env vars: BASE_MODEL, SFT_CKPT, AUDIO_ROOT, CURRICULUM
# Optional: NPROC (default 8), CONFIG (default configs/grpo_choice.yaml)
set -euo pipefail

NPROC="${NPROC:-8}"
CONFIG="${CONFIG:-configs/grpo_choice.yaml}"

: "${BASE_MODEL:?export BASE_MODEL}"
: "${SFT_CKPT:?export SFT_CKPT}"
: "${AUDIO_ROOT:?export AUDIO_ROOT}"
: "${CURRICULUM:?export CURRICULUM}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

torchrun --nproc_per_node="$NPROC" rl/train_grpo.py "$CONFIG"
