#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m experiments.exp4_ablation.evaluate \
  --exp3-dir Results/exp3/full_2026-07-05 \
  --out-dir Results/exp4/full_2026-07-05
