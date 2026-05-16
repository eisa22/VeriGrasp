"""Convert pipeline masks (closed_matches or SAM3D output) to CandidateOut objects."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from perception.candidate import CandidateOut
from perception.geometry.plane import heights_above_plane, project_to_plane_xy

FX = FY = 437.04


@dataclass
class MatchNeighbor:
    """A potential neighbour parcel extracted from a Stage-5 match.

    Contains the parcel's top height (above pallet) and its XY footprint in
    the pallet plane frame. Used by Stage 2.5 as a deterministic source of
    'next parcel below', including matches that were later discarded by the
    overlap dedup step.
    """
    match_id: str
    label: str
    status: str                                 # "kept" | "excluded_by_dedup"
    top_surface_height: float                   # height above pallet (m)
    footprint_xy: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    center_xy: np.ndarray
    occluded_by: str | None = None              # label of the kept match that hid us


def _footprint_from_xy(xy: np.ndarray) -> tuple[float, float, float, float]:
    if xy.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    )


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


def build_match_neighbors(
    matches: list[dict],
    depth: np.ndarray,
    plane_model: np.ndarray,
    status: str,
) -> list[MatchNeighbor]:
    """
    Build MatchNeighbor entries from Stage-5 matches (kept or excluded).

    Each match must have:
      - "mask" : HxW binary mask (the parcel's top surface)
      - "matched_box" : [x1, y1, x2, y2] (pixel coords)
      - "label"
      - "z_stats"["z_plane_m"] : top height above pallet (m), already
        in pallet-relative coords because Sobel runs with pallet_relative=True

    We re-backproject the mask to get the 3D footprint in plane-XY for a
    consistent neighbour search.
    """
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    plane = tuple(float(x) for x in plane_model)

    out: list[MatchNeighbor] = []
    for m in matches:
        mask = np.asarray(m["mask"], dtype=np.uint8)
        pts = _backproject_mask(mask, depth, cx, cy)
        if len(pts) < 10:
            continue
        heights = heights_above_plane(pts, plane)
        xy = project_to_plane_xy(pts, plane)
        top_h = float(np.percentile(heights, 95))
        fp = _footprint_from_xy(xy)
        center = np.array(
            [(fp[0] + fp[2]) * 0.5, (fp[1] + fp[3]) * 0.5], dtype=np.float64
        )
        mid = str(_stable_candidate_id(m.get("label", "match"), list(m["matched_box"]), top_h))
        out.append(
            MatchNeighbor(
                match_id=mid,
                label=str(m.get("label", "match")),
                status=status,
                top_surface_height=top_h,
                footprint_xy=fp,
                center_xy=center,
                occluded_by=m.get("_occluded_by"),
            )
        )
    return out


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
