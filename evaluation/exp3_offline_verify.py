"""Experiment 3: soft-score reconstruction from persisted per-check margins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

CHECK_ORDER: tuple[str, ...] = (
    "existence",
    "top_height_match",
    "bbox_extent",
    "bbox_inlier",
    "bbox_surface_dist",
    "bbox_top_normal",
    "bbox_coverage",
    "planarity",
    "normal_angle",
    "normal_scatter",
    "suction_area",
    "edge_clearance",
    "data_gaps",
    "depth_seam",
    "surface_warp",
    "normal_alignment",
    "surface_warp_robust",
    "suction_force",
    "wrench_lever",
    "corridor_clear",
)

# Copied from verification/config.py soft_score defaults (Exp-3 fix, no pipeline import).
SOFT_SCORE_WEIGHTS: dict[str, float] = {
    "existence": 1.0,
    "top_height_match": 1.5,
    "bbox_extent": 1.5,
    "bbox_inlier": 1.0,
    "bbox_surface_dist": 1.0,
    "bbox_top_normal": 1.0,
    "bbox_coverage": 1.0,
    "planarity": 2.0,
    "normal_angle": 2.0,
    "normal_scatter": 1.0,
    "suction_area": 1.5,
    "edge_clearance": 1.5,
    "data_gaps": 1.0,
    "depth_seam": 1.5,
    "surface_warp": 1.5,
    "normal_alignment": 1.5,
    "surface_warp_robust": 1.5,
    "suction_force": 2.0,
    "wrench_lever": 1.5,
    "corridor_clear": 2.0,
}

SOFT_SCORE_SCALES: dict[str, float] = {
    "existence": 0.4,
    "top_height_match": 0.03,
    "bbox_extent": 0.15,
    "bbox_inlier": 0.20,
    "bbox_surface_dist": 0.02,
    "bbox_top_normal": 12.0,
    "bbox_coverage": 0.60,
    "planarity": 0.0025,
    "normal_angle": 30.0,
    "normal_scatter": 0.2,
    "suction_area": 1.0,
    "edge_clearance": 0.02,
    "data_gaps": 0.08,
    "depth_seam": 0.40,
    "surface_warp": 0.008,
    "normal_alignment": 30.0,
    "surface_warp_robust": 0.006,
    "suction_force": 2.0,
    "wrench_lever": 0.5,
    "corridor_clear": 0.15,
}


@dataclass(frozen=True)
class CheckRecordFlat:
    name: str
    passed: bool
    margin: float
    unverifiable: bool


def soft_score_config() -> dict:
    return {"weights": SOFT_SCORE_WEIGHTS, "scales": SOFT_SCORE_SCALES}


def compute_soft_score_from_checks(
    checks: Mapping[str, CheckRecordFlat],
    cfg: dict | None = None,
    *,
    exclude: set[str] | None = None,
) -> float:
    sc = cfg if cfg is not None else soft_score_config()
    if "soft_score" in sc:
        sc = sc["soft_score"]
    weights = sc.get("weights", SOFT_SCORE_WEIGHTS)
    scales = sc.get("scales", SOFT_SCORE_SCALES)

    num = 0.0
    den = 0.0
    for name in CHECK_ORDER:
        if exclude and name in exclude:
            continue
        rec = checks.get(name)
        if rec is None:
            continue
        w = float(weights.get(name, 1.0))
        scale = float(scales.get(name, 1.0)) or 1.0
        norm = float(rec.margin) / scale
        norm = max(-3.0, min(3.0, norm))
        num += w * norm
        den += w
    return num / den if den > 0 else 0.0


def compute_soft_score_from_row(
    row: Mapping[str, object],
    cfg: dict | None = None,
    *,
    exclude: set[str] | None = None,
) -> float:
    checks: dict[str, CheckRecordFlat] = {}
    for name in CHECK_ORDER:
        margin = row.get(f"check_{name}_margin")
        if margin == "" or margin is None:
            continue
        passed = row.get(f"check_{name}_pass")
        uv = row.get(f"check_{name}_unverifiable")
        checks[name] = CheckRecordFlat(
            name=name,
            passed=bool(passed) if passed not in ("", None) else False,
            margin=float(margin),
            unverifiable=bool(uv) if uv not in ("", None) else False,
        )
    merged_cfg = soft_score_config()
    if cfg is not None:
        merged_cfg = cfg
    return compute_soft_score_from_checks(checks, merged_cfg, exclude=exclude)
