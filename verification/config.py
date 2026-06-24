"""Load verification config with documented code defaults.

All thresholds live here (and in verification.yaml) so the sensitivity
analysis (RQ4) can sweep them without touching code. Citations / norm mapping
notes are kept inline so the audit record is traceable to ISO/IEC TR 5469 and
the EU AI Act risk-control requirements.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Default approach axis: strictly vertical world-up. For a top-down camera the
# world-up direction is -Z in the OpenCV camera frame (toward the camera), which
# is also the direction Open3D normals are oriented to in the grasp backend.
_DEFAULT_CONFIG: dict[str, Any] = {
    # "cascade": stop at first failed check (fast, production default).
    # "full": always compute every check (fills all margins for ROC / RQ2).
    "mode": "cascade",
    "approach_axis": [0.0, 0.0, -1.0],
    # Documented for reproducibility; the plane fit is fully deterministic and
    # does not actually consume the seed (no RANSAC).
    "seed": 0,
    # Rectangular suction gripper centred on the grasp point (not a separate
    # suction pose). Width/length lie in the pallet-plane frame (u, v).
    "gripper": {
        # Standard 50 x 50 mm flat vacuum gripper (common parcel-picking pad).
        "width_m": 0.050,
        "length_m": 0.050,
        # Vertical clearance the gripper needs above the top face (lift path).
        "approach_height_m": 0.150,
        # Extra margin around the footprint for the Stage-3 lift corridor.
        "safety_margin_m": 0.005,
    },
    "stage1": {
        # Segmentation-vs-raw-cloud consistency gate.
        # Minimum fraction of mask pixels that carry valid depth; below this the
        # segmentation is not backed by real geometry (hallucinated / over holes).
        "min_valid_fraction": 0.60,
        # Absolute minimum number of valid mask points.
        "min_points": 50,
        # Planar radius of the measurement window around the grasp point. The
        # top height is measured here so it matches where the lift happens.
        "window_radius_m": 0.03,
        # Minimum points inside that window; fewer => fail-closed (grasp in hole).
        "min_window_points": 10,
        # Robust percentile of the local heights used as the top / lift height
        # (high percentile rejects flying-pixel outliers without taking the max).
        "top_percentile": 95.0,
        # Allowed deviation between the locally measured top height and the
        # pipeline-reported candidate.top_surface_height.
        "top_height_tol_m": 0.03,
    },
    "box_check": {
        # Deterministic, visibility-aware OBB plausibility check (Stage 1).
        # Min fraction of near-box points contained in the box (+eps).
        "inlier_min": 0.80,
        # Max median point-to-surface distance for near-surface points.
        "surface_dist_median_max_m": 0.02,
        # Max relative deviation of the robust per-axis span vs the box edge.
        "extent_rel_dev_max": 0.15,
        # Max angle between the PCA top-face normal and the box top normal.
        "top_normal_angle_max_deg": 12.0,
        # Below this many near-box points the box is UNVERIFIABLE.
        "min_points_total": 300,
        # Per expected-visible face: coverage below this -> UNVERIFIABLE.
        "min_coverage": 0.60,
        # Containment margin (m) added to the half extents for the inlier test.
        "eps_m": 0.01,
        # Band (m) defining "near the box surface / face".
        "near_face_band_m": 0.03,
        # Coverage raster per face (grid x grid cells).
        "coverage_grid": 10,
        # A face is expected-visible only if its outward normal faces the sensor
        # by at least this cosine (generalises the strict normal.view<0 rule).
        "face_visible_min_facing": 0.2,
    },
    "stage2": {
        # Max planarity RMSE of the gripper window (a few mm compliance).
        "plane_rmse_max_m": 0.0025,
        # Max tilt between surface normal and approach axis.
        "normal_angle_max_deg": 30.0,
        # Max scatter of per-point normals inside the window (rugged surface).
        "normal_scatter_max": 0.20,
        # Required contiguous flat coverage of the gripper footprint (1.0 = 100%).
        # Slightly below 1.0 to tolerate genuine sensor gaps at the footprint rim.
        "min_area_ratio": 0.95,
        # Grasp centre must not overhang the parcel edge by more than this
        # (defaults to min(half_width, half_length) when null).
        "edge_clearance_min_m": None,
        # Raster cell size for the contiguous-area / edge analysis. Must be
        # >= depth sampling (~depth/fx ≈ 5 mm here) or interior cells stay empty.
        "raster_m": 0.008,
        # Max fraction of footprint raster cells without any depth points.
        "max_empty_cell_fraction": 0.08,
        # Physical depth seam: neighbour cells must differ by less than this.
        "depth_seam_step_m": 0.015,
        # Max fraction of neighbour pairs along a row/column exceeding the step.
        "depth_seam_span_ratio": 0.40,
        # Max peak-to-valley height variation after plane fit (surface warp).
        "max_peak_to_valley_m": 0.008,
        # --- Additional physical checks (additive) ---
        # Max angle between SuctionNet grasp normal and fitted plane normal.
        "normal_alignment_max_deg": 30.0,
        # Outlier-robust warp: max percentile-spread of plane residuals.
        "max_peak_to_valley_robust_m": 0.006,
        "warp_robust_low_pct": 2.5,
        "warp_robust_high_pct": 97.5,
        # Holdability (suction wrench resistance) physical parameters.
        "object_density_kg_m3": 150.0,
        "vacuum_pressure_pa": 40000.0,
        "holdability_safety_factor": 2.0,
        "robust_fit": {
            "max_iter": 5,
            # Inlier band = mad_scale * MAD of residuals (deterministic).
            "mad_scale": 2.5,
            "min_points": 12,
        },
    },
    "corridor": {
        # Optional expansion on the parcel footprint when building the lift corridor
        # in the pipeline (before verification). 0 = exact package widest extent.
        "safety_margin_m": 0.0,
    },
    "stage3": {
        # Stage 3 only runs a raw-cloud collision test against the precomputed
        # extraction corridor (pipeline). Horizontal extents are NOT derived here.
        # Vertical extent of the safety lift corridor above the package top
        # (used when building the corridor in the pipeline).
        "safety_corridor_height_m": 0.30,
        # Deprecated alias — used only when safety_corridor_height_m is absent.
        "approach_height_m": None,
        # Points strictly above the top face inside the corridor up to this many
        # are tolerated as sensor noise (must be > tolerance to reject).
        "noise_point_tolerance": 5,
        # Points within this height of z_top are considered part of the grasp
        # surface, not obstacles.
        "top_band_m": 0.005,
    },
    "soft_score": {
        # Per-check weights for the combined soft score (RQ3). Keys are check
        # names; missing checks default to weight 1.0.
        "weights": {
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
        },
        # Margins are normalised by these scales (per check, in the check's unit)
        # before weighting. Missing checks default to 1.0.
        "scales": {
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
        },
        # Final soft verdict threshold: accept when score >= this.
        "accept_threshold": 0.0,
    },
}

_CONFIG_PATH = Path(__file__).resolve().parent / "verification.yaml"


@dataclass(frozen=True)
class GripperFootprint:
    """Rectangular gripper centred on the grasp point (pallet-plane frame)."""

    width_m: float
    length_m: float
    half_w_m: float
    half_l_m: float
    approach_height_m: float
    safety_margin_m: float


def resolve_corridor_height(cfg: dict[str, Any]) -> float:
    """Vertical safety-corridor height above the grasp top (metres).

    Priority: stage3.safety_corridor_height_m → stage3.approach_height_m
    (legacy) → gripper.approach_height_m.
    """
    s3 = cfg["stage3"]
    if s3.get("safety_corridor_height_m") is not None:
        return float(s3["safety_corridor_height_m"])
    approach_h = s3.get("approach_height_m")
    if approach_h is not None:
        return float(approach_h)
    return resolve_gripper(cfg).approach_height_m


def resolve_gripper(cfg: dict[str, Any]) -> GripperFootprint:
    """Read gripper dimensions from config (centre = grasp point)."""
    g = cfg["gripper"]
    width = float(g["width_m"])
    length = float(g["length_m"])
    return GripperFootprint(
        width_m=width,
        length_m=length,
        half_w_m=0.5 * width,
        half_l_m=0.5 * length,
        approach_height_m=float(g["approach_height_m"]),
        safety_margin_m=float(g["safety_margin_m"]),
    )


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_verification_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load verification config from YAML, merged over documented defaults."""
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    config_path = Path(path) if path is not None else _CONFIG_PATH
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, loaded)
    return cfg
