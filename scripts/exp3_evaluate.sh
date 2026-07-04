#!/usr/bin/env bash
# Experiment 3 — offline evaluation (no GPU / detector needed).
#
#   ./scripts/exp3_evaluate.sh [--out-dir PATH] [--visibility strict|primary|both]
#                              [--limit N] [--scenes ids] [--secondary-sample]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

cd "$REPO_ROOT"
exec "$PYTHON" -m experiments.exp3_verification.evaluate "$@"
