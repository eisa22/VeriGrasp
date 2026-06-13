"""Kamera-Intrinsics für Blender-RGB-D-Szenen (dataset_meta / ground_truth)."""

from __future__ import annotations

import json
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def load_dataset_meta(data_root: Path) -> dict:
    meta_path = data_root / "dataset_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_data_root(session_path: str | Path) -> Path:
    """Datensatz-Root mit dataset_meta.json (Parent von scene_*)."""
    path = Path(session_path).expanduser().resolve()
    if (path / "dataset_meta.json").exists():
        return path
    return path.parent


def load_camera_intrinsics(
    session_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, float | int]:
    """
    Lädt Pinhole-Intrinsics aus dataset_meta.json (Priorität) oder ground_truth.json.

    Returns:
        dict mit fx, fy, cx, cy, width, height
    """
    if data_root is None:
        if session_path is None:
            data_root = _project_root() / "Data" / "blender_dataset"
        else:
            data_root = resolve_data_root(session_path)
    else:
        data_root = Path(data_root).expanduser().resolve()

    meta = load_dataset_meta(data_root)
    cam = meta.get("camera") if meta else None

    if not cam and session_path is not None:
        gt_path = Path(session_path).expanduser().resolve() / "ground_truth.json"
        if gt_path.exists():
            with open(gt_path, encoding="utf-8") as f:
                cam = json.load(f).get("camera", {})

    if not cam:
        raise FileNotFoundError(
            f"Kamera-Intrinsics nicht gefunden in {data_root} "
            f"(dataset_meta.json oder ground_truth.json)"
        )

    return {
        "fx": float(cam["fx"]),
        "fy": float(cam["fy"]),
        "cx": float(cam["cx"]),
        "cy": float(cam["cy"]),
        "width": int(cam["width"]),
        "height": int(cam["height"]),
    }


def load_camera_height_m(session_path: str | Path, data_root: str | Path | None = None) -> float:
    """Welt-Z der Kamera (Höhe über Boden) aus ground_truth.json."""
    scene_dir = Path(session_path).expanduser().resolve()
    gt_path = scene_dir / "ground_truth.json"
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            cam = json.load(f).get("camera", {})
        loc = cam.get("location_world")
        if loc and len(loc) == 3:
            return float(loc[2])
        if "height_above_pallet_top" in cam:
            return float(cam["height_above_pallet_top"]) + 0.144

    if data_root is None:
        data_root = resolve_data_root(session_path)
    meta_cam = load_dataset_meta(Path(data_root)).get("camera", {})
    if "height_above_pallet_top_m" in meta_cam:
        return float(meta_cam["height_above_pallet_top_m"]) + 0.144

    return 2.644
