"""Experiment 2: per-candidate error metrics (mm / degrees).

Exact definitions from the thesis protocol (sec:exp:grasp-acc):
  e_lat    = || Pi(c_pred) - Pi(c_gt) ||_2                     [mm]
  e_top    = h_pred_top - h_gt_top                (signed)     [mm]
  e_bottom = h_pred_bottom - h_gt_bottom          (signed)     [mm]
  theta    = arccos(clamp(n_pred . n_gt, -1, 1))               [deg]
  extent   = (pred - gt) / gt for long / short / height        [rel]
  yaw      = folded (180 deg; 90 deg for near-square GT)       [deg]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.exp2_geometry import PredCandidateGeometry, yaw_error_deg
from evaluation.exp2_gt import GtObjectGeometry

M_TO_MM = 1000.0

# neighbor_source / bottom_method -> (cue name, thesis priority; tab:grasp:bottom)
BOTTOM_CUE_BY_SOURCE = {
    "match": ("overlap_matched_parcel", 1),
    "overlap": ("overlap_candidate", 2),
    "lateral": ("lateral_neighbor", 3),
    "gradient_global": ("gradient_global", 4),
    "gradient": ("gradient_ring", 5),
    "histogram": ("histogram_ring", 6),
    "scene_plane": ("scene_plane", 7),
}
BOTTOM_CUE_BY_METHOD = {
    "measured": ("measured_visible", 8),
    "from_pallet": ("fallback_pallet", 9),
    "uncertain": ("fallback_uncertain", 10),
}


def bottom_cue(bottom_method: str | None, neighbor_source: str | None) -> tuple[str, int]:
    """Map the stage-8 record to the thesis cue name and priority."""
    if bottom_method == "from_neighbor" and neighbor_source in BOTTOM_CUE_BY_SOURCE:
        return BOTTOM_CUE_BY_SOURCE[neighbor_source]
    if bottom_method in BOTTOM_CUE_BY_METHOD:
        return BOTTOM_CUE_BY_METHOD[bottom_method]
    return ("unknown", 99)


@dataclass
class CandidateErrors:
    """All Experiment 2 errors for one matched candidate."""

    e_lat_mm: float
    e_top_mm_signed: float
    ext_err_long_rel: float | None
    ext_err_short_rel: float | None
    ext_err_height_rel: float | None
    yaw_err_deg: float | None
    yaw_fold_deg: int | None
    e_bottom_mm_signed: float | None
    bottom_cue: str
    bottom_priority: int
    bottom_confidence: float | None


def candidate_errors(
    pred: PredCandidateGeometry,
    gt: GtObjectGeometry,
) -> CandidateErrors:
    e_lat = float(np.linalg.norm(pred.centroid_xy - gt.center_xy)) * M_TO_MM
    e_top = (pred.h_top - gt.h_top) * M_TO_MM

    if pred.footprint_long_m is not None:
        ext_long = (pred.footprint_long_m - gt.footprint_long_m) / gt.footprint_long_m
        ext_short = (pred.footprint_short_m - gt.footprint_short_m) / gt.footprint_short_m
        gt_h = max(gt.height_m, 1e-9)
        ext_height = (pred.height_m - gt.height_m) / gt_h if pred.height_m is not None else None
        yaw_err, yaw_fold = yaw_error_deg(
            pred.footprint_yaw_deg, gt.footprint_yaw_deg, gt.footprint_aspect
        )
    else:
        ext_long = ext_short = ext_height = None
        yaw_err = yaw_fold = None

    e_bottom = (
        (pred.h_bottom - gt.h_bottom) * M_TO_MM if pred.h_bottom is not None else None
    )
    cue, priority = bottom_cue(pred.bottom_method, pred.neighbor_source)

    return CandidateErrors(
        e_lat_mm=e_lat,
        e_top_mm_signed=e_top,
        ext_err_long_rel=ext_long,
        ext_err_short_rel=ext_short,
        ext_err_height_rel=ext_height,
        yaw_err_deg=yaw_err,
        yaw_fold_deg=yaw_fold,
        e_bottom_mm_signed=e_bottom,
        bottom_cue=cue,
        bottom_priority=priority,
        bottom_confidence=pred.bottom_confidence,
    )
