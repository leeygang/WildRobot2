#!/usr/bin/env bash
set -euo pipefail

WR2_SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
cd "$WR2_SCRIPT_DIR/.."
exec uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_xy_physics.yaml \
  --restore-checkpoint results/wr2_walking/wr2_xy_20261012_140728_seed0/checkpoints/000200007680.pt \
  "$@"
