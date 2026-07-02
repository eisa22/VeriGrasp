"""Ensure Experiment 1 runs with the project virtualenv (has torch, etc.)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_venv_python() -> Path | None:
    candidate = project_root() / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def reexec_with_project_venv() -> None:
    """Re-launch current script with .venv/bin/python if torch is missing."""
    vpy = project_venv_python()
    if vpy is None:
        _print_venv_help()
        sys.exit(1)
    print(f"[EXP1] Wechsle zu Projekt-Python: {vpy}")
    os.execv(str(vpy), [str(vpy), *sys.argv])


def _print_venv_help() -> None:
    root = project_root()
    print(
        "\n[EXP1] FEHLER: 'torch' nicht gefunden.\n"
        f"  Aktives Python: {sys.executable} ({sys.version.split()[0]})\n"
        f"  Dieses Projekt nutzt die venv unter: {root / '.venv'}\n\n"
        "  Loesung (im Repo-Root):\n"
        "    source .venv/bin/activate\n"
        "    python -m experiments.exp1_seg.run_inference --test-set smoke --gui\n\n"
        "  Oder ohne activate:\n"
        f"    {root / '.venv' / 'bin' / 'python'} -m experiments.exp1_seg.run_inference ...\n",
        file=sys.stderr,
    )


def require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        vpy = project_venv_python()
        if vpy is not None and Path(sys.executable).resolve() != vpy.resolve():
            reexec_with_project_venv()
        _print_venv_help()
        sys.exit(1)
