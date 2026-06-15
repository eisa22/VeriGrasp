"""Extraction lift corridor (pipeline Stage 12b).

The corridor cross-section matches the parcel footprint at its widest extent
(from ``parcel_obb``). Vertical extent is the safety lift height above the
package top. Verification (Stage 13) only tests this precomputed volume
against the raw scene point cloud.
"""

from __future__ import annotations

import numpy as np

from perception.candidate import CandidateOut
from perception.geometry.plane import plane_basis, project_to_plane_xy, unproject_from_plane_xy


def _orient_rot(long_dir_xy: np.ndarray | None) -> np.ndarray:
    if long_dir_xy is None:
        return np.eye(2, dtype=np.float64)
    d = np.asarray(long_dir_xy, dtype=np.float64).reshape(2)
    d = d / (np.linalg.norm(d) + 1e-12)
    return np.array([[d[0], -d[1]], [d[1], d[0]]], dtype=np.float64)


def _long_axis_in_plane(
    parcel_obb: dict,
    plane: tuple[float, float, float, float],
) -> np.ndarray | None:
    try:
        R = np.asarray(parcel_obb["R"], dtype=np.float64)
        ext = np.asarray(parcel_obb["extents"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    _, u, v = plane_basis(plane)
    d0 = np.array([float(R[:, 0] @ u), float(R[:, 0] @ v)])
    d1 = np.array([float(R[:, 1] @ u), float(R[:, 1] @ v)])
    long_dir = d0 if float(ext[0]) >= float(ext[1]) else d1
    norm = np.linalg.norm(long_dir)
    if norm < 1e-9:
        return None
    return long_dir / norm


def _footprint_corners_plane_xy(
    center_xy: np.ndarray,
    half_long: float,
    half_short: float,
    long_dir_xy: np.ndarray | None,
) -> np.ndarray:
    corners_g = np.array(
        [
            [-half_long, -half_short],
            [half_long, -half_short],
            [half_long, half_short],
            [-half_long, half_short],
        ],
        dtype=np.float64,
    )
    M = _orient_rot(long_dir_xy)
    return np.asarray(center_xy, dtype=np.float64).reshape(1, 2) + corners_g @ M.T


def _corridor_box_endpoints(
    center_cam: np.ndarray,
    plane: tuple[float, float, float, float],
    z_bottom_h: float,
    lift_height_m: float,
    half_long: float,
    half_short: float,
    long_dir_xy: np.ndarray | None = None,
) -> dict:
    center = np.asarray(center_cam, dtype=np.float64).reshape(1, 3)
    c_xy = project_to_plane_xy(center, plane)[0]
    corners_xy = _footprint_corners_plane_xy(c_xy, half_long, half_short, long_dir_xy)
    z_bot = float(z_bottom_h)
    z_top = z_bot + float(lift_height_m)
    bottom = unproject_from_plane_xy(corners_xy, plane, heights=z_bot)
    top = unproject_from_plane_xy(corners_xy, plane, heights=z_top)
    long_list = (
        [float(x) for x in long_dir_xy.tolist()] if long_dir_xy is not None else None
    )
    return {
        "z_bottom_m": z_bot,
        "corridor_z_top_m": z_top,
        "half_long_m": float(half_long),
        "half_short_m": float(half_short),
        "safety_corridor_height_m": float(lift_height_m),
        "corners_bottom_3d": [list(map(float, p)) for p in bottom],
        "corners_top_3d": [list(map(float, p)) for p in top],
        "long_dir_xy": long_list,
    }


def _mask_footprint_halves(
    candidate: CandidateOut,
    plane: tuple[float, float, float, float],
    safety_margin_m: float,
) -> tuple[np.ndarray, float, float, np.ndarray | None]:
    """Fallback when no parcel OBB: axis-aligned mask extent in the plane."""
    points = np.asarray(candidate.points_3d, dtype=np.float64)
    if len(points) < 3:
        c = np.asarray(candidate.centroid_3d, dtype=np.float64).reshape(1, 3)
        center = c[0]
        half = 0.05 + safety_margin_m
        return center, half, half, None
    xy = project_to_plane_xy(points, plane)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    c_xy = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
    ext_u = float(x1 - x0)
    ext_v = float(y1 - y0)
    half_long = 0.5 * max(ext_u, ext_v) + safety_margin_m
    half_short = 0.5 * min(ext_u, ext_v) + safety_margin_m
    center = unproject_from_plane_xy(
        c_xy.reshape(1, 2), plane, heights=float(candidate.top_surface_height)
    )[0]
    return center, half_long, half_short, None


def compute_extraction_corridor(
    candidate: CandidateOut,
    plane: tuple[float, float, float, float],
    lift_height_m: float,
    safety_margin_m: float = 0.0,
) -> dict | None:
    """Build the vertical extraction corridor for the selected parcel.

    Cross-section = parcel footprint at widest extent (``parcel_obb`` horizontal
  extents). Centred on the parcel OBB centre. Returns ``None`` when geometry
    cannot be determined.
    """
    z_bottom = float(candidate.top_surface_height)
    bottom = getattr(candidate, "bottom", None)
    parcel_obb = getattr(bottom, "parcel_obb", None) if bottom else None

    if parcel_obb:
        ext = np.asarray(parcel_obb["extents"], dtype=np.float64)
        half_long = 0.5 * float(max(ext[0], ext[1])) + float(safety_margin_m)
        half_short = 0.5 * float(min(ext[0], ext[1])) + float(safety_margin_m)
        center = np.asarray(parcel_obb["center"], dtype=np.float64).reshape(3)
        long_dir_xy = _long_axis_in_plane(parcel_obb, plane)
        source = "parcel_obb"
    else:
        center, half_long, half_short, long_dir_xy = _mask_footprint_halves(
            candidate, plane, safety_margin_m
        )
        source = "mask_footprint"

    if half_long <= 0 or half_short <= 0:
        return None

    box = _corridor_box_endpoints(
        center,
        plane,
        z_bottom,
        lift_height_m,
        half_long,
        half_short,
        long_dir_xy,
    )
    return {
        **box,
        "center_3d": list(map(float, center.reshape(3))),
        "corridor_half_long_m": half_long,
        "corridor_half_short_m": half_short,
        "package_top_m": z_bottom,
        "source": source,
    }
