"""Mask-driven grasp centroid zone (anchor + adaptive radius)."""

from __future__ import annotations

from typing import Any

import numpy as np

from perception.candidate import CandidateOut
from perception.grasp_generation.camera import CameraInfo, depth_to_point_cloud


def _pixel_to_3d(row: float, col: float, depth_m: float, camera: CameraInfo) -> np.ndarray:
    x = (col - camera.cx) * depth_m / camera.fx
    y = (row - camera.cy) * depth_m / camera.fy
    return np.array([x, y, depth_m], dtype=np.float64)


def _median_depth_in_window(
    depth: np.ndarray,
    mask: np.ndarray,
    row: int,
    col: int,
    window_px: int,
) -> float | None:
    h, w = depth.shape[:2]
    half = window_px // 2
    r0 = max(0, row - half)
    r1 = min(h, row + half + 1)
    c0 = max(0, col - half)
    c1 = min(w, col + half + 1)
    patch = depth[r0:r1, c0:c1]
    mpatch = mask[r0:r1, c0:c1]
    valid = patch[(patch > 0) & mpatch]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _mask_centroid_px(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    return int(round(float(ys.mean()))), int(round(float(xs.mean())))


def compute_anchor_3d(
    candidate: CandidateOut,
    mask: np.ndarray,
    depth: np.ndarray,
    camera: CameraInfo,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mask centroid + median depth window → anchor in camera frame."""
    cc = cfg.get("centroid_constraint") or {}
    window = int(cc.get("anchor_window_px", 7))
    meta: dict[str, Any] = {"anchor_source": "mask_centroid"}

    if not np.any(mask):
        anchor = np.asarray(candidate.centroid_3d, dtype=np.float64)
        meta["anchor_source"] = "candidate_centroid_fallback"
        return anchor, meta

    row, col = _mask_centroid_px(mask)
    z = _median_depth_in_window(depth, mask, row, col, window)
    if z is not None and z > 0:
        anchor = _pixel_to_3d(float(row), float(col), z, camera)
        meta["anchor_pixel"] = [row, col]
        meta["anchor_depth_m"] = z
        return anchor, meta

    if candidate.centroid_3d is not None and np.all(np.isfinite(candidate.centroid_3d)):
        anchor = np.asarray(candidate.centroid_3d, dtype=np.float64)
        meta["anchor_source"] = "candidate_centroid_3d"
        return anchor, meta

    if "center_xy" in candidate.debug:
        xy = np.asarray(candidate.debug["center_xy"], dtype=np.float64).reshape(-1)[:2]
        z_fb = float(np.median(depth[mask & (depth > 0)])) if np.any(mask & (depth > 0)) else 1.0
        anchor = np.array([xy[0], xy[1], z_fb], dtype=np.float64)
        meta["anchor_source"] = "center_xy_fallback"
        return anchor, meta

    anchor = _pixel_to_3d(float(row), float(col), 1.0, camera)
    meta["anchor_source"] = "default_depth_fallback"
    return anchor, meta


def compute_adaptive_radius_m(
    mask: np.ndarray,
    depth: np.ndarray,
    anchor_3d: np.ndarray,
    camera: CameraInfo,
    cfg: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Radius from percentile of 3D XY distances of mask pixels to anchor."""
    cc = cfg.get("centroid_constraint") or {}
    percentile = float(cc.get("distance_percentile", 70))
    scale = float(cc.get("radius_scale", 0.85))
    r_min = float(cc.get("min_radius_m", 0.03))
    r_max = float(cc.get("max_radius_m", 0.15))
    use_xy = bool(cc.get("use_xy_distance", True))

    point_cloud = depth_to_point_cloud(depth, camera)
    valid = mask & (depth > 0)
    if not np.any(valid):
        return r_min, {"radius_source": "min_fallback", "n_mask_pixels": 0}

    pts = point_cloud[valid]
    anchor = np.asarray(anchor_3d, dtype=np.float64)
    if use_xy:
        diff = pts[:, :2] - anchor[:2]
        dists = np.linalg.norm(diff, axis=1)
    else:
        dists = np.linalg.norm(pts - anchor, axis=1)

    if dists.size == 0:
        return r_min, {"radius_source": "min_fallback", "n_mask_pixels": 0}

    raw = float(np.percentile(dists, percentile))
    radius_m = float(np.clip(scale * raw, r_min, r_max))
    return radius_m, {
        "radius_source": "mask_percentile",
        "radius_percentile_raw_m": raw,
        "distance_percentile": percentile,
        "n_mask_pixels": int(valid.sum()),
        "dist_median_m": float(np.median(dists)),
        "dist_max_m": float(np.max(dists)),
    }


def build_centroid_disk_mask(
    point_cloud: np.ndarray,
    anchor_3d: np.ndarray,
    radius_m: float,
    use_xy_distance: bool = True,
) -> np.ndarray:
    """H×W bool mask: pixels within radius of anchor in 3D (XY or full 3D)."""
    anchor = np.asarray(anchor_3d, dtype=np.float64)
    if use_xy_distance:
        diff = point_cloud[..., :2] - anchor[:2]
        dist = np.linalg.norm(diff, axis=-1)
    else:
        diff = point_cloud - anchor
        dist = np.linalg.norm(diff, axis=-1)
    valid = point_cloud[..., 2] > 0
    return (dist <= radius_m) & valid


def pick_grasp_nearest_centroid(
    grasps: list,
    anchor_3d: np.ndarray,
    use_xy_distance: bool = True,
) -> tuple[object | None, int | None]:
    """Return grasp with smallest 3D XY (or 3D) distance to anchor."""
    if not grasps:
        return None, None
    anchor = np.asarray(anchor_3d, dtype=np.float64).reshape(3)
    best_idx = 0
    best_dist = float("inf")
    for i, g in enumerate(grasps):
        pos = np.asarray(g.position, dtype=np.float64).reshape(3)
        if use_xy_distance:
            dist = float(np.linalg.norm(pos[:2] - anchor[:2]))
        else:
            dist = float(np.linalg.norm(pos - anchor))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return grasps[best_idx], best_idx


def compute_grasp_centroid_zone(
    candidate: CandidateOut,
    mask: np.ndarray,
    depth: np.ndarray,
    camera: CameraInfo,
    cfg: dict[str, Any],
) -> tuple[np.ndarray | None, float | None, dict[str, Any]]:
    """Return anchor_3d, radius_m, debug dict; None,None if constraint disabled."""
    cc = cfg.get("centroid_constraint") or {}
    if not cc.get("enabled", False):
        return None, None, {"centroid_constraint_enabled": False}

    anchor, anchor_meta = compute_anchor_3d(candidate, mask, depth, camera, cfg)
    radius_m, radius_meta = compute_adaptive_radius_m(mask, depth, anchor, camera, cfg)
    debug = {
        "centroid_constraint_enabled": True,
        "anchor_3d": anchor.tolist(),
        "radius_m": radius_m,
        **anchor_meta,
        **radius_meta,
    }
    return anchor, radius_m, debug
