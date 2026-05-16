"""Pallet plane fitting and 2D/3D projection helpers."""

from __future__ import annotations

import numpy as np


def _normalize_plane(plane: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = plane
    plane_arr = np.array([a, b, c, d], dtype=np.float64)
    if plane_arr[2] < 0:
        plane_arr = -plane_arr
    return plane_arr


def plane_basis(plane: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit normal n and orthonormal tangent axes (u, v) on the plane."""
    plane_arr = _normalize_plane(plane)
    n = plane_arr[:3]
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / n_norm

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return n, u, v


def fit_pallet_plane(scene_pcd: np.ndarray) -> tuple[float, float, float, float]:
    """
    RANSAC plane fit on scene point cloud.

    Returns:
        (a, b, c, d) with positive Z component on the normal.
    """
    import open3d as o3d

    points = np.asarray(scene_pcd, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("scene_pcd must be (N, 3)")
    if len(points) < 3:
        return (0.0, 0.0, 1.0, 0.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000,
    )
    plane_arr = _normalize_plane(tuple(plane_model))
    return tuple(float(x) for x in plane_arr)


def project_to_plane_xy(
    points: np.ndarray,
    plane: tuple[float, float, float, float],
) -> np.ndarray:
    """Project 3D points onto pallet plane; return (N, 2) coordinates in plane frame."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    plane_arr = _normalize_plane(plane)
    n, u, v = plane_basis(tuple(plane_arr))
    a, b, c, d = plane_arr
    norm = np.sqrt(a * a + b * b + c * c) + 1e-12

    signed = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm
    projected = points - signed[:, None] * n[None, :]
    origin = -d / norm * n
    rel = projected - origin[None, :]
    xy = np.stack([rel @ u, rel @ v], axis=1)
    return xy


def unproject_from_plane_xy(
    xy: np.ndarray,
    plane: tuple[float, float, float, float],
    heights: float | np.ndarray | None = None,
) -> np.ndarray:
    """
    Map plane-frame (x, y) back to 3D camera frame.

    `heights` follows depth_rel convention: positive = closer to camera
    (opposite of the plane normal direction, which points away from camera).
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim == 1:
        xy = xy.reshape(1, 2)
    plane_arr = _normalize_plane(plane)
    n, u, v = plane_basis(tuple(plane_arr))
    a, b, c, d = plane_arr
    norm = np.sqrt(a * a + b * b + c * c) + 1e-12
    origin = -d / norm * n

    pts = origin[None, :] + xy[:, 0:1] * u[None, :] + xy[:, 1:2] * v[None, :]
    if heights is not None:
        h = np.asarray(heights, dtype=np.float64)
        if h.ndim == 0:
            h = np.full(len(xy), float(h))
        pts = pts - h[:, None] * n[None, :]
    return pts


def heights_above_plane(
    points: np.ndarray,
    plane: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Signed distance from plane, in depth_rel convention.

    Positive = above pallet (closer to camera in typical setup).
    Returns shape (N,).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0,), dtype=np.float64)
    plane_arr = _normalize_plane(plane)
    a, b, c, d = plane_arr
    norm = np.sqrt(a * a + b * b + c * c) + 1e-12
    signed = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm
    return -signed


def plane_z_at_xy(
    plane: tuple[float, float, float, float],
    x: float,
    y: float,
    z_hint: float | None = None,
) -> float:
    """Camera-frame Z on the pallet plane at (x, y)."""
    plane_arr = _normalize_plane(plane)
    a, b, c, d = plane_arr
    if abs(c) < 1e-6:
        return float(z_hint) if z_hint is not None else 0.0
    return float(-(a * x + b * y + d) / c)


def mad(values: np.ndarray | list[float]) -> float:
    """Median absolute deviation."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))
