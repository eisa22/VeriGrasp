"""Oriented bounding box fit and vertical extrusion for parcel volume."""

from __future__ import annotations

import numpy as np

from perception.candidate import CandidateOut
from perception.geometry.plane import plane_basis, project_to_plane_xy, unproject_from_plane_xy


def _axis_aligned_footprint_xy(
    candidate: CandidateOut,
    plane: tuple[float, float, float, float],
) -> np.ndarray:
    points = np.asarray(candidate.points_3d, dtype=np.float64)
    if len(points) >= 3:
        return project_to_plane_xy(points, plane)
    c = np.asarray(candidate.centroid_3d, dtype=np.float64).reshape(1, 3)
    return project_to_plane_xy(c, plane)


def fit_extruded_obb(
    candidate: CandidateOut,
    plane: tuple[float, float, float, float],
    bottom_z: float,
    top_z: float,
    config: dict,
) -> dict:
    """
    Fit OBB on top-surface projection and extrude to bottom_z.

    Returns dict with center, extents, R, corners_3d (8, 3).
    """
    import open3d as o3d

    obb_cfg = config.get("obb", {})
    min_points = int(obb_cfg.get("min_points", 50))
    max_aspect = float(obb_cfg.get("max_aspect_ratio", 20.0))

    points = np.asarray(candidate.points_3d, dtype=np.float64)
    xy = project_to_plane_xy(points, plane) if len(points) >= 3 else _axis_aligned_footprint_xy(candidate, plane)

    use_axis_aligned = len(xy) < min_points
    n, u, v = plane_basis(plane)
    center2d = None
    ext2d = None
    R2d = None

    if not use_axis_aligned:
        pts2d = np.zeros((len(xy), 3), dtype=np.float64)
        pts2d[:, 0] = xy[:, 0]
        pts2d[:, 1] = xy[:, 1]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts2d)
        try:
            obb2d = pcd.get_oriented_bounding_box()
            ext2d = np.asarray(obb2d.extent, dtype=np.float64)
            ratio = float(ext2d.max() / (ext2d.min() + 1e-9))
            if ratio > max_aspect:
                use_axis_aligned = True
            else:
                center2d = np.asarray(obb2d.center, dtype=np.float64)
                R2d = np.asarray(obb2d.R, dtype=np.float64)
        except Exception:
            use_axis_aligned = True

    if use_axis_aligned:
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        center2d = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, 0.0], dtype=np.float64)
        ext2d = np.array([max(x1 - x0, 1e-4), max(y1 - y0, 1e-4), 1e-6], dtype=np.float64)
        R2d = np.eye(3, dtype=np.float64)

    height = max(abs(float(top_z) - float(bottom_z)), 1e-3)
    center_top = unproject_from_plane_xy(
        center2d[:2].reshape(1, 2), plane, heights=top_z
    )[0]
    center_bottom = unproject_from_plane_xy(
        center2d[:2].reshape(1, 2), plane, heights=bottom_z
    )[0]
    center_3d = 0.5 * (center_top + center_bottom)

    Ru = R2d[:, 0]
    Rv = R2d[:, 1]
    R_world = np.column_stack([
        Ru[0] * u + Ru[1] * v,
        Rv[0] * u + Rv[1] * v,
        n,
    ])
    extents = np.array([ext2d[0], ext2d[1], height], dtype=np.float64)

    half = extents * 0.5
    local_corners = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    corners_3d = center_3d[None, :] + (local_corners * half[None, :]) @ R_world.T

    return {
        "center": center_3d.tolist(),
        "extents": extents.tolist(),
        "R": R_world.tolist(),
        "corners_3d": corners_3d.tolist(),
    }


def obb_xy_footprint_area(parcel_obb: dict) -> float:
    ext = np.asarray(parcel_obb["extents"], dtype=np.float64)
    return float(ext[0] * ext[1])
