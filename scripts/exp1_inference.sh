#!/usr/bin/env bash
# Experiment 1 — always uses project .venv (torch, transformers, …)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[exp1] Keine .venv gefunden. Bitte im Repo-Root: uv venv && uv pip install -r …" >&2
  exit 1
fi
exec "$PY" -m experiments.exp1_seg.run_inference "$@"
