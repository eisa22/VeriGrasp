"""Global flat-surface detection for Stage-2.5 bottom inference.

Scans the workspace once for every flat region that is NOT already
covered by a detected SAM3D candidate. The resulting `ScenePlane`
objects serve as:
  - extra reference surfaces in `infer_bottom_planes` (4th neighbour
    source besides gradient / depth-histogram / lateral candidates)
  - input for the Stage-9 visualization so the user can see every plane
    the pipeline considers
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from perception.bottom_inference.neighbors import FX, FY
from perception.geometry.plane import heights_above_plane


@dataclass
class ScenePlane:
    """A flat reference surface detected anywhere in the workspace."""
    plane_id: int
    height_above_pallet: float            # robust median height (m)
    height_std_m: float
    area_px: int
    area_m2: float
    aspect_ratio: float
    centroid_px: tuple[float, float]
    centroid_xy_cam: tuple[float, float]  # mean XY in camera frame
    footprint_xy_cam: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) in cam-XY
    mask: np.ndarray = field(repr=False)  # full-size bool mask, for viz
    points_3d: np.ndarray = field(repr=False)  # backprojected (camera frame)


def _backproject(ys: np.ndarray, xs: np.ndarray, depth: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / FX
    y = (ys.astype(np.float64) - cy) * z / FY
    return np.stack([x, y, z], axis=1)


def detect_scene_planes(
    depth: np.ndarray,
    sobel_edges: np.ndarray | None,
    workspace_mask: np.ndarray | None,
    exclude_masks: list[np.ndarray] | None,
    plane: tuple[float, float, float, float],
    config: dict,
) -> list[ScenePlane]:
    """Detect ALL flat reference surfaces in the workspace.

    Algorithm:
      1. Start from `workspace_mask & (depth > 0)`.
      2. Subtract the union of all `exclude_masks` (already-detected
         SAM3D candidates), dilated slightly so border pixels of those
         objects don't leak in as flat-surface samples.
      3. Subtract Sobel/Canny edges (dilated) so we operate inside
         coherent flat regions only.
      4. Connected-components on the remaining mask.
      5. Per CC, build a height histogram. Every histogram bin with
         enough population is a candidate height-band; each band's
         largest connected sub-region becomes a `ScenePlane`. This
         splits CCs that happen to contain multiple height levels
         (e.g. when one CC fuses pallet and box-side-wall pixels).
      6. Filter every band by `min_area_m2`, `min_aspect`,
         `max_height_std_m`.
    """
    cfg = config.get("scene_planes", {})
    if not cfg.get("enabled", True):
        return []

    edge_dilate_px = int(cfg.get("edge_dilate_px", 2))
    exclude_dilate_px = int(cfg.get("exclude_dilate_px", 4))
    min_component_px = int(cfg.get("min_component_px", 200))
    min_area_m2 = float(cfg.get("min_area_m2", 0.003))
    min_aspect = float(cfg.get("min_aspect_ratio", 0.15))
    max_height_std_m = float(cfg.get("max_height_std_m", 0.020))
    slab_m = float(cfg.get("slab_m", 0.005))
    min_band_pixels = int(cfg.get("min_band_pixels", 100))
    min_band_fraction = float(cfg.get("min_band_fraction", 0.05))

    depth_arr = np.asarray(depth, dtype=np.float64)
    H, W = depth_arr.shape

    if workspace_mask is not None:
        search = np.asarray(workspace_mask, dtype=bool).copy()
    else:
        search = np.ones((H, W), dtype=bool)
    search &= depth_arr > 0

    if exclude_masks:
        union = np.zeros((H, W), dtype=np.uint8)
        for m in exclude_masks:
            if m is None:
                continue
            union |= (np.asarray(m, dtype=np.uint8) > 0).astype(np.uint8)
        if exclude_dilate_px > 0 and union.any():
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (2 * exclude_dilate_px + 1, 2 * exclude_dilate_px + 1),
            )
            union = cv2.dilate(union, k, iterations=1)
        search &= ~(union > 0)

    if sobel_edges is not None:
        edges = np.asarray(sobel_edges, dtype=np.uint8) > 0
        if edge_dilate_px > 0 and edges.any():
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (2 * edge_dilate_px + 1, 2 * edge_dilate_px + 1),
            )
            edges = cv2.dilate(edges.astype(np.uint8), k, iterations=1) > 0
        search &= ~edges

    if not search.any():
        return []

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        search.astype(np.uint8), connectivity=8,
    )

    out: list[ScenePlane] = []
    next_id = 0

    for li in range(1, n_labels):
        cc_area_px = int(stats[li, cv2.CC_STAT_AREA])
        if cc_area_px < min_component_px:
            continue

        ys, xs = np.where(labels == li)
        pts = _backproject(ys, xs, depth_arr)
        heights = heights_above_plane(pts, plane)
        n_cc = heights.size

        h_min, h_max = float(heights.min()), float(heights.max())
        if h_max - h_min < slab_m:
            sub_bins = [(slice(None),)]
            edges_h = np.array([h_min, h_max + slab_m])
            hist = np.array([n_cc])
        else:
            n_bins = max(1, int(np.ceil((h_max - h_min) / slab_m)))
            hist, edges_h = np.histogram(heights, bins=n_bins)

        threshold = max(min_band_pixels, int(min_band_fraction * n_cc))

        for bi in range(len(hist)):
            if hist[bi] < threshold:
                continue
            lo, hi = float(edges_h[bi]), float(edges_h[bi + 1])
            in_band = (heights >= lo) & (heights <= hi)
            if not in_band.any():
                continue

            band_mask_full = np.zeros((H, W), dtype=np.uint8)
            band_mask_full[ys[in_band], xs[in_band]] = 1

            n_sl, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(
                band_mask_full, connectivity=8,
            )
            if n_sl <= 1:
                continue

            for sli in range(1, n_sl):
                sub_area_px = int(sub_stats[sli, cv2.CC_STAT_AREA])
                if sub_area_px < min_component_px:
                    continue
                w_px = int(sub_stats[sli, cv2.CC_STAT_WIDTH])
                h_px = int(sub_stats[sli, cv2.CC_STAT_HEIGHT])
                aspect = min(w_px, h_px) / max(max(w_px, h_px), 1)
                if aspect < min_aspect:
                    continue

                sub_ys, sub_xs = np.where(sub_labels == sli)
                sub_pts = _backproject(sub_ys, sub_xs, depth_arr)
                sub_heights = heights_above_plane(sub_pts, plane)
                z_std = float(np.std(sub_heights))
                if z_std > max_height_std_m:
                    continue
                mean_zcam = float(np.median(sub_pts[:, 2]))
                area_m2 = sub_area_px * (mean_zcam / FX) * (mean_zcam / FY)
                if area_m2 < min_area_m2:
                    continue

                sub_mask = np.zeros((H, W), dtype=bool)
                sub_mask[sub_ys, sub_xs] = True

                out.append(ScenePlane(
                    plane_id=next_id,
                    height_above_pallet=float(np.median(sub_heights)),
                    height_std_m=z_std,
                    area_px=sub_area_px,
                    area_m2=area_m2,
                    aspect_ratio=aspect,
                    centroid_px=(float(sub_xs.mean()), float(sub_ys.mean())),
                    centroid_xy_cam=(float(sub_pts[:, 0].mean()),
                                     float(sub_pts[:, 1].mean())),
                    footprint_xy_cam=(
                        float(sub_pts[:, 0].min()),
                        float(sub_pts[:, 1].min()),
                        float(sub_pts[:, 0].max()),
                        float(sub_pts[:, 1].max()),
                    ),
                    mask=sub_mask,
                    points_3d=sub_pts,
                ))
                next_id += 1

    return out


def _xy_overlap(
    fp_a: tuple[float, float, float, float],
    fp_b: tuple[float, float, float, float],
) -> float:
    """Min-area fraction of the intersection in XY."""
    ax0, ay0, ax1, ay1 = fp_a
    bx0, by0, bx1, by1 = fp_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-9, (bx1 - bx0) * (by1 - by0))
    return inter / max(min(area_a, area_b), 1e-9)


@dataclass
class ScenePlaneNeighborResult:
    z_highest_neighbor: float | None
    chosen_plane_id: int | None
    qualifying_ids: list[int]
    rejection_counts: dict


def find_neighbor_via_scene_planes(
    target_points_xy: np.ndarray,
    z_visible_min: float,
    scene_planes: list[ScenePlane],
    config: dict,
) -> ScenePlaneNeighborResult:
    """Pick the highest scene plane that 'supports' the target.

    A plane qualifies when:
      - height_above_pallet < z_visible_min - tol
      - XY footprint overlaps with target's XY footprint
        (min-area overlap >= `min_overlap`) OR the plane centroid
        lies within `max_centroid_dist_m` of the target centroid.
    """
    cfg = config.get("scene_planes", {})
    tol = float(config.get("tolerance_m", 0.008))
    min_overlap = float(cfg.get("min_overlap", 0.05))
    max_centroid_dist_m = float(cfg.get("max_centroid_dist_m", 0.20))

    if not scene_planes or target_points_xy.size == 0:
        return ScenePlaneNeighborResult(None, None, [], {})

    tgt_fp = (
        float(target_points_xy[:, 0].min()),
        float(target_points_xy[:, 1].min()),
        float(target_points_xy[:, 0].max()),
        float(target_points_xy[:, 1].max()),
    )
    tgt_center = (float(target_points_xy[:, 0].mean()),
                  float(target_points_xy[:, 1].mean()))

    rej = {"too_high": 0, "no_overlap": 0}
    qualifying: list[tuple[float, int]] = []

    for p in scene_planes:
        if p.height_above_pallet >= z_visible_min - tol:
            rej["too_high"] += 1
            continue

        overlap = _xy_overlap(tgt_fp, p.footprint_xy_cam)
        dx = p.centroid_xy_cam[0] - tgt_center[0]
        dy = p.centroid_xy_cam[1] - tgt_center[1]
        dist = float(np.hypot(dx, dy))

        if overlap < min_overlap and dist > max_centroid_dist_m:
            rej["no_overlap"] += 1
            continue

        qualifying.append((p.height_above_pallet, p.plane_id))

    if not qualifying:
        return ScenePlaneNeighborResult(None, None, [], rej)

    qualifying.sort(key=lambda x: -x[0])
    best_h, best_id = qualifying[0]
    return ScenePlaneNeighborResult(
        z_highest_neighbor=best_h,
        chosen_plane_id=best_id,
        qualifying_ids=[i for _, i in qualifying],
        rejection_counts=rej,
    )
