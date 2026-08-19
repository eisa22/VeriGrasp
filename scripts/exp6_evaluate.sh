#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
"$PYTHON" scripts/patch_exp3_n_blocking.py \
  --csv Results/exp3/full_2026-07-05/exp3_per_grasp.csv \
  --data-root Data/blender_dataset
"$PYTHON" -m experiments.exp6_sensitivity.evaluate \
  --exp3-dir Results/exp3/full_2026-07-05 \
  --out-dir Results/exp6/full_2026-07-05
