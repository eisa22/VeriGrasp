"""Reproducibility snapshot for experiment runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from config import DINO_MODEL_ID, PROJECT_ROOT


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def config_hash(config_dict: dict[str, Any]) -> str:
    blob = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_env_snapshot(device: str, extra: dict | None = None) -> dict:
    snap = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "dino_model_id": DINO_MODEL_ID,
        "git_commit": _git_commit(),
        "packages": _package_versions(
            ["torch", "numpy", "opencv-python", "transformers", "open3d", "scikit-learn"]
        ),
    }
    if extra:
        snap.update(extra)
    return snap


def write_env_snapshot(run_dir: Path, device: str, config_dict: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    snap = build_env_snapshot(device, {"config_hash": config_hash(config_dict)})
    path = run_dir / "env.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    return path


def write_config_snapshot(run_dir: Path, config_dict: dict) -> Path:
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config_snapshot.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=True)
    return path


def load_pipeline_config() -> dict:
    """Serialize relevant config.py thresholds."""
    import config as cfg

    keys = [
        k
        for k in dir(cfg)
        if k.isupper() and not k.startswith("_")
    ]
    out: dict[str, Any] = {}
    for k in sorted(keys):
        v = getattr(cfg, k)
        if isinstance(v, (str, int, float, bool, list, tuple)):
            out[k] = v
        elif isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
    return out
