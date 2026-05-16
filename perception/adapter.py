"""Convert pipeline masks (closed_matches or SAM3D output) to CandidateOut objects."""

from __future__ import annotations

import hashlib

import numpy as np

from perception.candidate import CandidateOut
from perception.geometry.plane import heights_above_plane

FX = FY = 437.04


def _stable_candidate_id(label: str, bbox: list, top_h: float) -> str:
    raw = f"{label}|{bbox}|{round(top_h, 4)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _backproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    cx: float,
    cy: float,
) -> np.ndarray:
    ys, xs = np.where((mask > 0) & (depth > 0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / FX
    y = (ys.astype(np.float64) - cy) * z / FY
    return np.stack([x, y, z], axis=1)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def build_scene_pcd_from_depth(
    depth: np.ndarray,
    workspace_mask: np.ndarray | None = None,
    stride: int = 4,
) -> np.ndarray:
    """Subsampled scene point cloud in camera coordinates."""
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    valid = depth > 0
    if workspace_mask is not None:
        valid &= workspace_mask
    ys, xs = np.where(valid)
    ys = ys[::stride]
    xs = xs[::stride]
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / FX
    y = (ys.astype(np.float64) - cy) * z / FY
    return np.stack([x, y, z], axis=1)


def _build_candidate(
    mask: np.ndarray,
    label: str,
    bbox: tuple[int, int, int, int],
    depth: np.ndarray,
    plane: tuple[float, float, float, float],
    cx: float,
    cy: float,
    surface_normal: np.ndarray,
) -> CandidateOut | None:
    """Common builder used by both closed_matches and SAM3D mask adapters."""
    points_3d = _backproject_mask(mask, depth, cx, cy)
    if len(points_3d) < 10:
        return None

    heights = heights_above_plane(points_3d, plane)
    top_h = float(np.percentile(heights, 95))
    centroid = points_3d.mean(axis=0)

    pixel_count = int((mask > 0).sum())
    mean_z = float(np.mean(points_3d[:, 2]))
    surface_area_m2 = pixel_count * (mean_z / FX) * (mean_z / FY)

    cid = _stable_candidate_id(label, list(bbox), top_h)

    return CandidateOut(
        candidate_id=cid,
        mask_2d=mask,
        points_3d=points_3d,
        centroid_3d=centroid,
        surface_normal=surface_normal.copy(),
        surface_area_m2=surface_area_m2,
        top_surface_height=top_h,
        bbox_2d=bbox,
        debug={"label": label},
    )


def _surface_normal_from_plane(plane: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, _ = plane
    norm = np.sqrt(a * a + b * b + c * c) + 1e-12
    return np.array([a, b, c], dtype=np.float64) / norm


def build_candidates_from_closed_matches(
    closed_matches: list[dict],
    depth: np.ndarray,
    plane_model: np.ndarray,
    session_context=None,
) -> list[CandidateOut]:
    """Build CandidateOut list from visualizer closed_matches entries."""
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    plane = tuple(float(x) for x in plane_model)
    surface_normal = _surface_normal_from_plane(plane)

    candidates: list[CandidateOut] = []
    for match in closed_matches:
        mask = np.asarray(match["mask"], dtype=np.uint8)
        bbox = tuple(int(v) for v in match["matched_box"])
        label = str(match.get("label", "parcel"))
        cand = _build_candidate(mask, label, bbox, depth, plane, cx, cy, surface_normal)
        if cand is not None:
            candidates.append(cand)

    return candidates


def build_candidates_from_sam3d(
    sam3d_masks: list[np.ndarray],
    sam3d_labels: list[str],
    depth: np.ndarray,
    plane_model: np.ndarray,
    sam3d_boxes: list | None = None,
    session_context=None,
) -> list[CandidateOut]:
    """Build CandidateOut list from SAM3D-refined masks."""
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    plane = tuple(float(x) for x in plane_model)
    surface_normal = _surface_normal_from_plane(plane)

    candidates: list[CandidateOut] = []
    for idx, (mask, label) in enumerate(zip(sam3d_masks, sam3d_labels)):
        mask = np.asarray(mask, dtype=np.uint8)
        if sam3d_boxes is not None and idx < len(sam3d_boxes) and sam3d_boxes[idx] is not None:
            bbox = tuple(int(v) for v in sam3d_boxes[idx])
        else:
            bbox = _mask_bbox(mask)
        cand = _build_candidate(mask, str(label), bbox, depth, plane, cx, cy, surface_normal)
        if cand is not None:
            candidates.append(cand)

    return candidates
