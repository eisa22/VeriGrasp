"""Geometry helpers for grasp verification.

All point clouds are in the OpenCV camera frame (x right, y down, z = depth).
Heights above the pallet and plane-frame XY use the shared helpers in
perception.geometry.plane so the convention matches the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.geometry.plane import plane_basis, project_to_plane_xy


def long_axis_in_plane(
    parcel_obb: dict | None,
    plane: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Unit 2D vector (pallet-plane u/v frame) of the parcel's longer side.

    Returns None when no OBB is available; callers then fall back to the
    axis-aligned u direction.
    """
    if not parcel_obb:
        return None
    try:
        R = np.asarray(parcel_obb["R"], dtype=np.float64)  # cols: axis0, axis1, n
        ext = np.asarray(parcel_obb["extents"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    _, u, v = plane_basis(plane)
    a0 = R[:, 0]
    a1 = R[:, 1]
    d0 = np.array([float(a0 @ u), float(a0 @ v)])
    d1 = np.array([float(a1 @ u), float(a1 @ v)])
    n0 = np.linalg.norm(d0)
    n1 = np.linalg.norm(d1)
    if n0 < 1e-9 and n1 < 1e-9:
        return None
    # Pick the in-plane axis with the larger extent as the parcel "long" side.
    long_dir = d0 if float(ext[0]) >= float(ext[1]) else d1
    norm = np.linalg.norm(long_dir)
    if norm < 1e-9:
        return None
    return long_dir / norm


def _gripper_rot(long_dir_xy: np.ndarray | None) -> np.ndarray:
    """2x2 matrix M with columns (long_dir, perp). plane = centre + g @ M.T."""
    if long_dir_xy is None:
        return np.eye(2, dtype=np.float64)
    d = np.asarray(long_dir_xy, dtype=np.float64).reshape(2)
    d = d / (np.linalg.norm(d) + 1e-12)
    return np.array([[d[0], -d[1]], [d[1], d[0]]], dtype=np.float64)


def gripper_corners_plane_xy(
    g_xy: np.ndarray,
    half_long: float,
    half_short: float,
    long_dir_xy: np.ndarray | None,
) -> np.ndarray:
    """Footprint corner coords (4, 2) in plane frame, centred on the grasp.

    Long side (half_long) is aligned with `long_dir_xy`; short side is
    perpendicular. Corner order: BL, BR, TR, TL in gripper frame.
    """
    corners_g = np.array(
        [
            [-half_long, -half_short],
            [half_long, -half_short],
            [half_long, half_short],
            [-half_long, half_short],
        ],
        dtype=np.float64,
    )
    M = _gripper_rot(long_dir_xy)
    return np.asarray(g_xy, dtype=np.float64).reshape(1, 2) + corners_g @ M.T


def corridor_box_endpoints_3d(
    p_g_cam: np.ndarray,
    plane: tuple[float, float, float, float],
    z_bottom_h: float,
    approach_h: float,
    half_long: float,
    half_short: float,
    long_dir_xy: np.ndarray | None = None,
) -> dict:
    """8 Eckpunkte des Entnahmekorridors in Kamera-Koordinaten.

    Der Korridor ist das Greifer-Rechteck (half_long × half_short) extrudiert
    von ``z_bottom_h`` (Paket-Oberfläche über Palette) bis ``z_bottom_h + approach_h``.
    """
    from perception.geometry.plane import project_to_plane_xy, unproject_from_plane_xy

    p_g = np.asarray(p_g_cam, dtype=np.float64).reshape(1, 3)
    g_xy = project_to_plane_xy(p_g, plane)[0]
    corners_xy = gripper_corners_plane_xy(g_xy, half_long, half_short, long_dir_xy)
    z_bot = float(z_bottom_h)
    z_corridor_top = z_bot + float(approach_h)
    bottom = unproject_from_plane_xy(corners_xy, plane, heights=z_bot)
    top = unproject_from_plane_xy(corners_xy, plane, heights=z_corridor_top)
    return {
        "z_bottom_m": z_bot,
        "corridor_z_top_m": z_corridor_top,
        "half_long_m": float(half_long),
        "half_short_m": float(half_short),
        "safety_corridor_height_m": float(approach_h),
        "corners_bottom_3d": [list(map(float, p)) for p in bottom],
        "corners_top_3d": [list(map(float, p)) for p in top],
    }


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
    half_long_m: float,
    half_short_m: float,
    plane: tuple[float, float, float, float],
    long_dir_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Points inside the rectangular gripper footprint centred on the grasp.

    The grasp position is the footprint centre (not a separate suction pose).
    When ``long_dir_xy`` is given the footprint is rotated so its long side
    (half_long_m) aligns with that pallet-plane direction; otherwise it is
    axis-aligned in the (u, v) frame.

    Returns (P_grip (K, 3), xy_local (K, 2) in the gripper frame: column 0 =
    along the long side, column 1 = along the short side).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    xy = project_to_plane_xy(points, plane)
    g_xy = project_to_plane_xy(np.asarray(p_g, dtype=np.float64).reshape(1, 3), plane)[0]
    rel = xy - g_xy[None, :]
    # Rotate plane-frame rel into the gripper frame (long axis -> x).
    M = _gripper_rot(long_dir_xy)
    rel_g = rel @ M
    inside = (
        (np.abs(rel_g[:, 0]) <= half_long_m)
        & (np.abs(rel_g[:, 1]) <= half_short_m)
    )
    return points[inside], rel_g[inside]


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
