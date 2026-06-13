"""Geometry helpers for grasp verification.

All point clouds are in the OpenCV camera frame (x right, y down, z = depth).
Heights above the pallet and plane-frame XY use the shared helpers in
perception.geometry.plane so the convention matches the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.geometry.plane import project_to_plane_xy


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_session(cls, session_context) -> "Intrinsics":
        return cls(
            fx=float(session_context.fx),
            fy=float(session_context.fy),
            cx=float(session_context.cx),
            cy=float(session_context.cy),
        )


def backproject(
    rows: np.ndarray,
    cols: np.ndarray,
    z: np.ndarray,
    intr: Intrinsics,
) -> np.ndarray:
    """Back-project pixel (row, col) + depth z to 3D camera-frame points."""
    x = (cols.astype(np.float64) - intr.cx) * z / intr.fx
    y = (rows.astype(np.float64) - intr.cy) * z / intr.fy
    return np.stack([x, y, z.astype(np.float64)], axis=1)


def full_pointcloud(depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """All valid (depth > 0) points of the scene as (N, 3) in camera frame."""
    rows, cols = np.where(depth > 0)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[rows, cols]
    return backproject(rows, cols, z, intr)


def target_pointcloud(
    depth: np.ndarray,
    mask: np.ndarray,
    intr: Intrinsics,
) -> np.ndarray:
    """Points on the target parcel only (mask > 0, depth > 0), camera frame."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != depth.shape[:2]:
        import cv2

        h, w = depth.shape[:2]
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    rows, cols = np.where(mask_bool & (depth > 0))
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[rows, cols]
    return backproject(rows, cols, z, intr)


def gather_bbox_points(
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    intr: Intrinsics,
) -> tuple[np.ndarray, int, int]:
    """Points whose pixel lies inside the bbox.

    Returns (P_bbox (M, 3) camera frame, n_valid, n_bbox_px).
    """
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 3), dtype=np.float64), 0, 0

    sub = depth[y1:y2, x1:x2]
    n_bbox_px = int(sub.size)
    rr, cc = np.where(sub > 0)
    if rr.size == 0:
        return np.zeros((0, 3), dtype=np.float64), 0, n_bbox_px
    z = sub[rr, cc]
    pts = backproject(rr + y1, cc + x1, z, intr)
    return pts, int(rr.size), n_bbox_px


def gather_gripper_points(
    points: np.ndarray,
    p_g: np.ndarray,
    half_w_m: float,
    half_l_m: float,
    plane: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Points inside the rectangular gripper footprint centred on the grasp.

    The grasp position is the footprint centre (not a separate suction pose).
    Footprint is axis-aligned in the pallet-plane (u, v) frame.

    Returns (P_grip (K, 3), xy_local (K, 2) relative to the grasp centre).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    xy = project_to_plane_xy(points, plane)
    g_xy = project_to_plane_xy(np.asarray(p_g, dtype=np.float64).reshape(1, 3), plane)[0]
    rel = xy - g_xy[None, :]
    inside = (
        (np.abs(rel[:, 0]) <= half_w_m)
        & (np.abs(rel[:, 1]) <= half_l_m)
    )
    return points[inside], rel[inside]


def robust_plane_fit(
    points: np.ndarray,
    max_iter: int = 5,
    mad_scale: float = 2.5,
    min_points: int = 12,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Deterministic robust plane fit (no RANSAC).

    PCA least-squares initial fit -> MAD-based inlier band on the residuals ->
    refit on inliers, repeated for a fixed number of iterations. Fully
    reproducible with explicit outlier handling.

    Returns (point_on_plane (3,), unit_normal (3,), rmse_inliers, inlier_mask).
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        centroid = points.mean(axis=0) if n else np.zeros(3)
        normal = np.array([0.0, 0.0, -1.0])
        mask = np.ones(n, dtype=bool)
        return centroid, normal, 0.0, mask

    inlier_mask = np.ones(n, dtype=bool)
    centroid = points.mean(axis=0)
    normal = np.array([0.0, 0.0, -1.0])

    for _ in range(max(1, int(max_iter))):
        pts = points[inlier_mask]
        if len(pts) < max(3, int(min_points) // 2):
            break
        centroid = pts.mean(axis=0)
        # PCA: normal = eigenvector of smallest eigenvalue of covariance.
        cov = np.cov((pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        residuals = np.abs((points - centroid) @ normal)
        med = float(np.median(residuals[inlier_mask]))
        mad = float(np.median(np.abs(residuals[inlier_mask] - med)))
        # 1.4826 scales MAD to a std estimate for Gaussian noise.
        sigma = 1.4826 * mad
        if sigma < 1e-9:
            band = max(med * 2.0, 1e-6)
        else:
            band = med + mad_scale * sigma
        new_mask = residuals <= band
        if new_mask.sum() < max(3, int(min_points)):
            # Keep the closest min_points to avoid collapsing the fit.
            order = np.argsort(residuals)
            new_mask = np.zeros(n, dtype=bool)
            new_mask[order[: max(3, int(min_points))]] = True
        if np.array_equal(new_mask, inlier_mask):
            inlier_mask = new_mask
            break
        inlier_mask = new_mask

    # Orient normal to point "up" (toward camera, -Z) for consistency.
    if normal[2] > 0:
        normal = -normal

    residuals = np.abs((points[inlier_mask] - centroid) @ normal)
    rmse = float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0
    return centroid, normal, rmse, inlier_mask


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two vectors (sign-agnostic, 0..90)."""
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 90.0
    cos = abs(float(np.dot(a, b) / (na * nb)))
    cos = min(1.0, max(-1.0, cos))
    return float(np.degrees(np.arccos(cos)))
