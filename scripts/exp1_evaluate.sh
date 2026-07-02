#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[exp1] Keine .venv gefunden." >&2
  exit 1
fi
exec "$PY" -m experiments.exp1_seg.evaluate "$@"
