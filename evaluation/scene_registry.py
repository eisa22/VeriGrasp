"""Scene index → evaluation category (Table 9.1b rows)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LIGHTING_PREFIX = "extreme_lighting_"

TOP_DOWN_ROWS = (
    "baseline",
    "mixed",
    "dense",
    "chaotic",
    "lighting",
    "occlusion",
    "edge",
    "tilted",
)

ANGLED_ROWS = (
    "angled_dense",
    "angled_chaotic",
    "angled_occluded",
)

ALL_CATEGORY_ROWS = TOP_DOWN_ROWS + ANGLED_ROWS

_SCENE_CATEGORY_MAP = {
    "chaotic_stack": "chaotic",
    "high_occlusion": "occlusion",
    "edge_cases": "edge",
    "tilted_pallet": "tilted",
    "angled_view_dense": "angled_dense",
    "angled_view_chaotic": "angled_chaotic",
    "angled_view_occluded": "angled_occluded",
}


@dataclass(frozen=True)
class SceneMeta:
    scene_id: str
    scene_index: int
    category: str
    viewpoint: str  # "top_down" | "angled"
    scene_category_raw: str | None


def scene_index_from_id(scene_id: str) -> int:
    return int(scene_id.replace("scene_", ""))


def load_gt_json(session_path: str | Path) -> dict:
    path = Path(session_path) / "ground_truth.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_angled(gt: dict) -> bool:
    cam = gt.get("camera") or {}
    return cam.get("view_tilt_from_vertical_deg") is not None


def category_for_scene(scene_id: str, gt: dict | None = None) -> str:
    idx = scene_index_from_id(scene_id)
    if gt is None:
        gt = {}
    raw = gt.get("scene_category")
    if raw:
        if raw.startswith(LIGHTING_PREFIX):
            return "lighting"
        if raw in _SCENE_CATEGORY_MAP:
            return _SCENE_CATEGORY_MAP[raw]
    if idx <= 55:
        return "baseline"
    if idx <= 199:
        return "mixed"
    if idx <= 349:
        return "dense"
    return "dense"


def scene_meta(session_path: str | Path) -> SceneMeta:
    session_path = Path(session_path)
    scene_id = session_path.name
    gt = load_gt_json(session_path)
    cat = category_for_scene(scene_id, gt)
    vp = "angled" if is_angled(gt) else "top_down"
    return SceneMeta(
        scene_id=scene_id,
        scene_index=scene_index_from_id(scene_id),
        category=cat,
        viewpoint=vp,
        scene_category_raw=gt.get("scene_category"),
    )
