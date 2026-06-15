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
        # Minimum fraction of valid depth pixels inside the bbox; below this the
        # height is not trustworthy (too many holes).
        "tau_valid": 0.60,
        # Height histogram bin size for top-cluster detection.
        "hist_bin_m": 0.005,
        # The top cluster must hold at least this fraction of bbox points to be
        # considered a dominant, unambiguous top face.
        "top_cluster_min_fraction": 0.35,
        # Two height plateaus separated by more than this gap => stacked /
        # adjacent objects inside one bbox.
        "plateau_gap_m": 0.015,
        # A continuous depth step of at least this size spanning the bbox is
        # treated as a seam between two side-by-side objects.
        "depth_edge_step_m": 0.020,
        # Fraction of a bbox row/column the seam must span to count as "through".
        "depth_edge_span_ratio": 0.70,
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
        "robust_fit": {
            "max_iter": 5,
            # Inlier band = mad_scale * MAD of residuals (deterministic).
            "mad_scale": 2.5,
            "min_points": 12,
        },
    },
    "stage3": {
        # Corridor half-extents default to gripper half-size + safety margin.
        "corridor_half_w_m": None,
        "corridor_half_l_m": None,
        # Vertical extent of the safety lift corridor above the grasp top face.
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
            "valid_ratio": 1.0,
            "top_cluster": 1.0,
            "single_object": 1.0,
            "no_seam": 1.0,
            "planarity": 2.0,
            "normal_angle": 2.0,
            "normal_scatter": 1.0,
            "suction_area": 1.5,
            "edge_clearance": 1.5,
            "corridor_clear": 2.0,
        },
        # Margins are normalised by these scales (per check, in the check's unit)
        # before weighting. Missing checks default to 1.0.
        "scales": {
            "valid_ratio": 0.4,
            "top_cluster": 0.5,
            "single_object": 0.05,
            "no_seam": 0.05,
            "planarity": 0.0025,
            "normal_angle": 30.0,
            "normal_scatter": 0.2,
            "suction_area": 1.0,
            "edge_clearance": 0.02,
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
