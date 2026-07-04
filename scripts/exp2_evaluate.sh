#!/usr/bin/env bash
# Experiment 2 — offline evaluation (no GPU needed).
#
#   ./scripts/exp2_evaluate.sh [--out-dir PATH] [--visibility primary|strict|both]
#                              [--limit N] [--scenes ids] [--gt-self-test]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

cd "$REPO_ROOT"
exec "$PYTHON" -m experiments.exp2_grasp.evaluate "$@"
