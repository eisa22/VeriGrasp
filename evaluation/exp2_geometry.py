"""Experiment 2: predicted-side derivations and angle/yaw helpers.

Predicted quantities come from the persisted ``stage8_candidates.json``
records; everything is expressed in the same pipeline pallet frame as the GT
derivations in :mod:`evaluation.exp2_gt`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.geometry.plane import plane_basis, project_to_plane_xy

NEAR_SQUARE_ASPECT = 1.2


@dataclass
class PredCandidateGeometry:
    """Pallet-frame geometry of one predicted candidate."""

    candidate_id: str
    centroid_xy: np.ndarray          # (2,) candidate centroid in plane frame
    h_top: float                     # 95th-percentile top height (m, pallet-rel.)
    h_bottom: float | None           # inferred OBB bottom (m, pallet-rel.)
    footprint_yaw_deg: float | None  # long-axis yaw in plane frame (deg)
    footprint_long_m: float | None
    footprint_short_m: float | None
    height_m: float | None           # OBB top - OBB bottom
    bottom_method: str | None
    bottom_confidence: float | None
    neighbor_source: str | None


def _obb_plane_yaw_and_extents(
    parcel_obb: dict,
    plane: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Yaw of the long footprint axis in the plane frame plus sorted extents."""
    R = np.asarray(parcel_obb["R"], dtype=np.float64)
    ext = np.asarray(parcel_obb["extents"], dtype=np.float64)
    _, u, v = plane_basis(plane)

    # Footprint axes are the first two OBB columns; pick the longer one.
    if ext[0] >= ext[1]:
        long_axis, long_e, short_e = R[:, 0], float(ext[0]), float(ext[1])
    else:
        long_axis, long_e, short_e = R[:, 1], float(ext[1]), float(ext[0])

    yaw = float(np.degrees(np.arctan2(float(long_axis @ v), float(long_axis @ u))))
    yaw = ((yaw + 90.0) % 180.0) - 90.0
    return yaw, long_e, short_e


def derive_pred_geometry(
    record: dict,
    plane: tuple[float, float, float, float],
) -> PredCandidateGeometry:
    """Derive pallet-frame quantities from one stage8 candidate record."""
    centroid = np.asarray(record["centroid_3d"], dtype=np.float64)
    centroid_xy = project_to_plane_xy(centroid.reshape(1, 3), plane)[0]

    h_top = float(record["top_surface_height_m"])
    h_bottom = record.get("bottom_z_m")
    h_bottom = float(h_bottom) if h_bottom is not None else None

    obb = record.get("parcel_obb")
    if obb:
        yaw, long_e, short_e = _obb_plane_yaw_and_extents(obb, plane)
        height = h_top - h_bottom if h_bottom is not None else None
    else:
        yaw = long_e = short_e = height = None

    return PredCandidateGeometry(
        candidate_id=str(record["candidate_id"]),
        centroid_xy=centroid_xy,
        h_top=h_top,
        h_bottom=h_bottom,
        footprint_yaw_deg=yaw,
        footprint_long_m=long_e,
        footprint_short_m=short_e,
        height_m=height,
        bottom_method=record.get("bottom_method"),
        bottom_confidence=record.get("bottom_confidence"),
        neighbor_source=record.get("neighbor_source"),
    )


def yaw_error_deg(
    psi_pred_deg: float,
    psi_gt_deg: float,
    gt_aspect: float,
    *,
    near_square_aspect: float = NEAR_SQUARE_ASPECT,
) -> tuple[float, int]:
    """Folded yaw error.

    e_yaw = min_k |psi_pred - psi_gt + k*fold|, folded into [0, fold/2].
    fold = 180 deg normally; 90 deg for near-square footprints (aspect < 1.2),
    where the long axis is not meaningful. Returns (error_deg, fold_deg).
    """
    fold = 90 if gt_aspect < near_square_aspect else 180
    d = (psi_pred_deg - psi_gt_deg) % fold
    err = min(d, fold - d)
    return float(err), fold


def angle_between_deg(n_a: np.ndarray, n_b: np.ndarray) -> float:
    """Angle between two unit vectors in degrees.

    Mathematically arccos(clamp(a . b, -1, 1)); implemented via
    atan2(||a x b||, a . b), which is numerically stable near 0 deg
    (arccos amplifies rounding error to sqrt(2*eps) there).
    """
    a = np.asarray(n_a, dtype=np.float64)
    b = np.asarray(n_b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    cross = float(np.linalg.norm(np.cross(a, b)))
    dot = float(a @ b)
    return float(np.degrees(np.arctan2(cross, dot)))


def orient_toward_camera(n: np.ndarray) -> np.ndarray:
    """Flip so the normal points toward the camera (n_z < 0, OpenCV +z forward)."""
    n = np.asarray(n, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[2] > 0:
        n = -n
    return n
