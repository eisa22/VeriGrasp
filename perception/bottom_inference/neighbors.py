"""Neighbour finding and solid-surface detection for bottom-plane inference."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from perception.candidate import CandidateOut
from perception.geometry.plane import heights_above_plane, project_to_plane_xy

from config import CAMERA_FX, CAMERA_FY

# Default intrinsics – kept in sync with config.CAMERA_FX/CAMERA_FY.
FX = CAMERA_FX
FY = CAMERA_FY


@dataclass
class CandidateGeometry:
    candidate_id: str
    center_xy: np.ndarray
    obb_extent_xy: float
    obb_footprint: tuple[float, float, float, float]
    top_surface_height: float
    z_visible_min: float


@dataclass
class NeighborInfo:
    neighbor_ids: list[str]
    neighbor_tops: list[float]
    neighbor_distances: list[float]
    z_neighbor_top: float | None
    z_highest_neighbor: float | None
    highest_neighbor_id: str | None
    neighbor_spread: float | None
    effective_radius_m: float = float("nan")


def _footprint_from_xy(xy: np.ndarray) -> tuple[float, float, float, float]:
    if xy.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    )


def _extent_from_footprint(fp: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = fp
    return max(x1 - x0, y1 - y0)


def _has_solid_2d_region(
    xy_points: np.ndarray,
    min_pixels: int,
    raster_m: float,
    min_aspect: float = 0.0,
) -> bool:
    """
    Rasterize points onto a small grid and check largest connected component.

    A 'solid' region must:
      - have at least `min_pixels` connected pixels
      - have a bounding-box aspect ratio >= `min_aspect` (rejects thin rings)
    """
    if len(xy_points) < min_pixels:
        return False
    x_min, y_min = xy_points.min(axis=0)
    x_max, y_max = xy_points.max(axis=0)
    width = max(1, int(np.ceil((x_max - x_min) / raster_m)) + 1)
    height = max(1, int(np.ceil((y_max - y_min) / raster_m)) + 1)
    if width * height < min_pixels:
        return False
    img = np.zeros((height, width), dtype=np.uint8)
    xs = np.clip(((xy_points[:, 0] - x_min) / raster_m).astype(int), 0, width - 1)
    ys = np.clip(((xy_points[:, 1] - y_min) / raster_m).astype(int), 0, height - 1)
    img[ys, xs] = 1
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    if n_labels <= 1:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas))
    max_area = int(areas[best])
    if max_area < min_pixels:
        return False
    if min_aspect > 0.0:
        w = int(stats[1 + best, cv2.CC_STAT_WIDTH])
        h = int(stats[1 + best, cv2.CC_STAT_HEIGHT])
        if min(w, h) / max(max(w, h), 1) < min_aspect:
            return False
    return True


def _solid_surface_params(config: dict) -> dict:
    cfg = config.get("solid_surface", {})
    return {
        "slab_m": float(cfg.get("slab_m", 0.005)),
        "min_points": int(cfg.get("min_points", 60)),
        "min_fraction": float(cfg.get("min_fraction", 0.05)),
        "min_component_pixels": int(cfg.get("min_component_pixels", 200)),
        "min_aspect": float(cfg.get("min_aspect_ratio", 0.0)),
        "raster_m": float(cfg.get("raster_m", 0.005)),
    }


def find_lowest_solid_surface(
    heights: np.ndarray,
    xy: np.ndarray,
    config: dict,
) -> float:
    """Lowest height-band that forms a solid (connected) surface.

    Walks bands from lowest to highest. First one with enough points AND a
    connected 2D patch (not a thin ring) wins. Falls back to the 5 %-percentile.
    """
    if len(heights) == 0:
        return 0.0
    p = _solid_surface_params(config)
    threshold = max(p["min_points"], int(np.ceil(p["min_fraction"] * len(heights))))

    h_min, h_max = float(heights.min()), float(heights.max())
    if h_max - h_min < p["slab_m"]:
        return float(np.median(heights))
    n_bins = max(1, int(np.ceil((h_max - h_min) / p["slab_m"])))
    hist, edges = np.histogram(heights, bins=n_bins)

    for i, count in enumerate(hist):
        if count < threshold:
            continue
        in_band = (heights >= edges[i]) & (heights <= edges[i + 1])
        if not _has_solid_2d_region(
            xy[in_band], p["min_component_pixels"], p["raster_m"], p["min_aspect"]
        ):
            continue
        return float(np.median(heights[in_band]))

    return float(np.percentile(heights, 5))


def find_highest_solid_surface(
    heights: np.ndarray,
    xy: np.ndarray,
    config: dict,
) -> float:
    """Highest height-band that forms a solid connected surface."""
    if len(heights) == 0:
        return 0.0
    p = _solid_surface_params(config)
    threshold = max(p["min_points"], int(np.ceil(p["min_fraction"] * len(heights))))

    h_min, h_max = float(heights.min()), float(heights.max())
    if h_max - h_min < p["slab_m"]:
        return float(np.median(heights))
    n_bins = max(1, int(np.ceil((h_max - h_min) / p["slab_m"])))
    hist, edges = np.histogram(heights, bins=n_bins)

    for i in range(len(hist) - 1, -1, -1):
        if hist[i] < threshold:
            continue
        in_band = (heights >= edges[i]) & (heights <= edges[i + 1])
        if not _has_solid_2d_region(
            xy[in_band], p["min_component_pixels"], p["raster_m"], p["min_aspect"]
        ):
            continue
        return float(np.median(heights[in_band]))

    return float(np.percentile(heights, 95))


def compute_candidate_geometry(
    candidate: CandidateOut,
    plane: tuple[float, float, float, float],
    config: dict,
) -> CandidateGeometry:
    points = np.asarray(candidate.points_3d, dtype=np.float64)
    if len(points) > 0:
        xy = project_to_plane_xy(points, plane)
        heights = heights_above_plane(points, plane)
        z_visible_min = find_lowest_solid_surface(heights, xy, config)
    else:
        c = np.asarray(candidate.centroid_3d, dtype=np.float64)
        xy = project_to_plane_xy(c.reshape(1, 3), plane)
        z_visible_min = float(candidate.top_surface_height)

    footprint = _footprint_from_xy(xy)
    extent = _extent_from_footprint(footprint)
    center_xy = np.array(
        [(footprint[0] + footprint[2]) * 0.5, (footprint[1] + footprint[3]) * 0.5],
        dtype=np.float64,
    )

    return CandidateGeometry(
        candidate_id=candidate.candidate_id,
        center_xy=center_xy,
        obb_extent_xy=extent,
        obb_footprint=footprint,
        top_surface_height=float(candidate.top_surface_height),
        z_visible_min=z_visible_min,
    )


def build_geometry_index(
    candidates: list[CandidateOut],
    plane: tuple[float, float, float, float],
    config: dict,
) -> dict[str, CandidateGeometry]:
    return {
        c.candidate_id: compute_candidate_geometry(c, plane, config)
        for c in candidates
    }


def _ring_mask(
    scene_xy: np.ndarray,
    footprint: tuple[float, float, float, float],
    outer_m: float,
    inner_m: float,
) -> np.ndarray:
    """
    Boolean mask: points inside `footprint + outer_m` but OUTSIDE
    `footprint + inner_m`. Forms a ring around the upper parcel.
    """
    x0, y0, x1, y1 = footprint
    in_outer = (
        (scene_xy[:, 0] >= x0 - outer_m)
        & (scene_xy[:, 0] <= x1 + outer_m)
        & (scene_xy[:, 1] >= y0 - outer_m)
        & (scene_xy[:, 1] <= y1 + outer_m)
    )
    in_inner = (
        (scene_xy[:, 0] >= x0 - inner_m)
        & (scene_xy[:, 0] <= x1 + inner_m)
        & (scene_xy[:, 1] >= y0 - inner_m)
        & (scene_xy[:, 1] <= y1 + inner_m)
    )
    return in_outer & ~in_inner


def find_neighbor_surface_via_scene(
    target: CandidateGeometry,
    scene_pcd: np.ndarray,
    plane: tuple[float, float, float, float],
    config: dict,
) -> tuple[float | None, int]:
    """
    Find the highest solid surface AROUND the parcel that lies strictly
    below the parcel's lowest visible surface.

    Search region is a RING around the OBB footprint:
       - outer boundary = footprint + `neighbor_ring_outer_m`
       - inner boundary = footprint + `neighbor_ring_inner_m`
    The inner rim is excluded because the parcel's own side walls project
    onto the footprint edge and would otherwise be detected as a 'solid'
    surface right below the top.

    If the ring yields no surface, fall back to the full (footprint +
    outer_m) area so single-parcel-on-pallet cases still work.

    Returns (z_surface or None, number_of_points_used).
    """
    if scene_pcd is None or len(scene_pcd) == 0:
        return None, 0

    cfg = config.get("solid_surface", {})
    min_step = float(cfg.get("min_step_m", 0.030))
    outer_m = float(cfg.get("neighbor_ring_outer_m", 0.10))
    inner_m = float(cfg.get("neighbor_ring_inner_m", 0.010))

    scene_xy = project_to_plane_xy(scene_pcd, plane)
    scene_heights = heights_above_plane(scene_pcd, plane)
    upper_bound = target.z_visible_min - min_step
    below = scene_heights < upper_bound

    # Primary: search ring around the parcel
    sel = _ring_mask(scene_xy, target.obb_footprint, outer_m, inner_m) & below
    if int(sel.sum()) > 0:
        z = find_highest_solid_surface(scene_heights[sel], scene_xy[sel], config)
        return z, int(sel.sum())

    # Fallback: full expanded footprint (solo parcel on pallet etc.)
    x0, y0, x1, y1 = target.obb_footprint
    in_fp = (
        (scene_xy[:, 0] >= x0 - outer_m)
        & (scene_xy[:, 0] <= x1 + outer_m)
        & (scene_xy[:, 1] >= y0 - outer_m)
        & (scene_xy[:, 1] <= y1 + outer_m)
    )
    sel = in_fp & below
    n_used = int(sel.sum())
    if n_used == 0:
        return None, 0
    z = find_highest_solid_surface(scene_heights[sel], scene_xy[sel], config)
    return z, n_used


def find_neighbor_via_matches(
    target: CandidateGeometry,
    match_neighbors: list,
    config: dict,
) -> tuple[float | None, str | None, str | None]:
    """
    Find a 'neighbour' match parcel whose top is strictly below the target's
    lowest visible surface. The neighbour must be in the spatial vicinity:

      1. Preferred: footprints overlap (IoU >= match_neighbor_min_iou).
      2. Fallback: 2D distance between footprint centers <= match_neighbor_max_dist_m.

    Among all qualifying neighbours, the HIGHEST top wins (= the surface
    closest to the target, i.e. the parcel the target is presumably resting
    on).

    User spec: 'compare the lowest point of the object with the neighbour
    object (even if it was excluded earlier). Pull the bounding box down
    UNLESS the lowest point is already at the same level or deeper toward
    z_pallet.'

    Returns (top_height, match_id, status) or (None, None, None).
    """
    if not match_neighbors:
        return None, None, None

    tol = float(config.get("tolerance_m", 0.008))
    min_iou = float(config.get("match_neighbor_min_iou", 0.05))
    max_dist = float(config.get("match_neighbor_max_dist_m", 0.40))
    z_lowest = float(target.z_visible_min)

    candidates: list[tuple[float, str, str, float, float]] = []  # (top, id, status, iou, dist)

    for n in match_neighbors:
        if getattr(n, "match_id", None) is None:
            continue
        n_h = float(n.top_surface_height)
        if n_h >= z_lowest - tol:
            continue
        iou = footprint_iou(target.obb_footprint, n.footprint_xy)
        dist = float(np.linalg.norm(target.center_xy - n.center_xy))
        if iou < min_iou and dist > max_dist:
            continue
        candidates.append((n_h, n.match_id, n.status, iou, dist))

    if not candidates:
        return None, None, None

    # Highest qualifying top wins.
    candidates.sort(key=lambda x: -x[0])
    best_h, best_id, best_status, best_iou, best_dist = candidates[0]
    print(
        f"[BOTTOM-NB] target z_min={z_lowest:.3f}m -> "
        f"picked {best_status} '{best_id[:8]}' top={best_h:.3f}m "
        f"(iou={best_iou:.2f}, d={best_dist*1000:.0f}mm, "
        f"pool considered={len(candidates)})"
    )
    return best_h, best_id, best_status


def find_lateral_neighbors(
    target: CandidateGeometry,
    all_geom: dict[str, CandidateGeometry],
    config: dict,
) -> NeighborInfo:
    """Detected-candidate neighbours: top-surfaces of OTHER parcels.

    A candidate qualifies as neighbour when its detected top is strictly
    below the target's lowest visible point (`z_visible_min`). The highest
    qualifying neighbour is the "next surface below the target" and pulls
    the bottom plane down to that height.
    """
    lateral_radius_m = float(config["lateral_radius_m"])
    lateral_radius_factor = float(config.get("lateral_radius_factor", 1.5))
    lateral_radius_max_m = float(config.get("lateral_radius_max_m", 1.5))
    tol = float(config.get("tolerance_m", config.get("height_tolerance", 0.008)))

    obb = float(target.obb_extent_xy)
    r_neighbor = max(lateral_radius_m, obb * lateral_radius_factor)
    r_neighbor = min(r_neighbor, lateral_radius_max_m)

    ids: list[str] = []
    tops: list[float] = []
    dists: list[float] = []

    others = [g for cid, g in all_geom.items() if cid != target.candidate_id]
    if not others:
        return NeighborInfo([], [], [], None, None, None, None, r_neighbor)

    centers = np.array([g.center_xy for g in others], dtype=np.float64)
    tree = cKDTree(centers)
    idxs = tree.query_ball_point(target.center_xy, r_neighbor)

    reference = float(target.z_visible_min)
    for i in idxs:
        g = others[i]
        if g.top_surface_height >= reference - tol:
            continue
        dist = float(np.linalg.norm(g.center_xy - target.center_xy))
        ids.append(g.candidate_id)
        tops.append(float(g.top_surface_height))
        dists.append(dist)

    order = np.argsort(ids)
    ids = [ids[i] for i in order]
    tops = [tops[i] for i in order]
    dists = [dists[i] for i in order]

    if not ids:
        return NeighborInfo([], [], [], None, None, None, None, r_neighbor)

    med = float(np.median(tops))
    spread = float(np.median(np.abs(np.array(tops) - med)))
    highest_idx = int(max(range(len(tops)), key=lambda i: tops[i]))

    return NeighborInfo(
        neighbor_ids=ids,
        neighbor_tops=tops,
        neighbor_distances=dists,
        z_neighbor_top=med,
        z_highest_neighbor=float(tops[highest_idx]),
        highest_neighbor_id=ids[highest_idx],
        neighbor_spread=spread,
        effective_radius_m=r_neighbor,
    )


@dataclass
class GradientPlateau:
    """Flat region separated from others by Sobel/Canny edges."""
    label: int
    area_px: int
    area_m2: float
    height_above_pallet: float
    height_std_m: float
    aspect_ratio: float
    centroid_px: tuple[float, float]
    global_id: int | None = None
    footprint_xy_cam: tuple[float, float, float, float] | None = None
    mask: np.ndarray | None = None
    points_3d: np.ndarray | None = None


@dataclass
class GradientNeighborResult:
    z_highest_neighbor: float | None
    chosen_label: int | None
    plateaus: list[GradientPlateau]
    n_ring_pixels: int
    n_components_total: int = 0
    rejection_counts: dict | None = None
    effective_radius_m: float = float("nan")
    radius_px: int = 0


def _dominant_height(heights: np.ndarray, slab_m: float) -> float:
    """Median height inside the histogram bin with the most points.

    More robust than np.percentile when a plateau's Z values are bi-modal
    (e.g. two height regimes glued together because Sobel missed an edge).
    """
    if heights.size == 0:
        return 0.0
    h_min, h_max = float(heights.min()), float(heights.max())
    if h_max - h_min < slab_m:
        return float(np.median(heights))
    n_bins = max(1, int(np.ceil((h_max - h_min) / slab_m)))
    hist, edges = np.histogram(heights, bins=n_bins)
    best = int(np.argmax(hist))
    in_band = (heights >= edges[best]) & (heights <= edges[best + 1])
    if not in_band.any():
        return float(np.median(heights))
    return float(np.median(heights[in_band]))


def _backproject_pixels(
    ys: np.ndarray,
    xs: np.ndarray,
    depth: np.ndarray,
    fx: float = FX,
    fy: float = FY,
) -> np.ndarray:
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def _build_neighborhood_ring(
    target_mask: np.ndarray,
    depth: np.ndarray,
    workspace_mask: np.ndarray | None,
    radius_min_m: float,
    adaptive_factor: float,
    max_radius_m: float,
    obb_extent_xy_m: float | None,
    fallback_radius_px: int,
) -> tuple[np.ndarray, float, int]:
    """Return (ring_mask, effective_radius_m, radius_px).

    Adaptive: r_m = clamp(max(radius_min_m, adaptive_factor * obb_extent), max_radius_m).
    Pixel radius derived via the parcel's mean depth.
    """
    if radius_min_m is None:
        return _empty_ring(target_mask, fallback_radius_px), float("nan"), fallback_radius_px

    target_depth = depth[target_mask]
    valid = target_depth[target_depth > 0]
    mean_depth = float(np.median(valid)) if valid.size > 0 else 1.0

    base = float(radius_min_m)
    if obb_extent_xy_m is not None and adaptive_factor > 0:
        effective_radius_m = max(base, float(obb_extent_xy_m) * adaptive_factor)
    else:
        effective_radius_m = base
    effective_radius_m = min(effective_radius_m, max_radius_m)

    radius_px = max(10, int(round(effective_radius_m * FX / max(mean_depth, 0.1))))
    return _empty_ring(target_mask, radius_px, workspace_mask, depth), effective_radius_m, radius_px


def _empty_ring(
    target_mask: np.ndarray,
    radius_px: int,
    workspace_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
    )
    dilated = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1) > 0
    ring = dilated & ~target_mask
    if workspace_mask is not None:
        ring &= np.asarray(workspace_mask, dtype=bool)
    if depth is not None:
        ring &= depth > 0
    return ring


def _plateau_filter_cfg(config: dict, section: str) -> dict:
    """Merge plateau-filter keys; global section falls back to gradient_neighbor."""
    cfg = dict(config.get(section, {}))
    base = config.get("gradient_neighbor", {})
    for key in (
        "min_plateau_area_px",
        "min_plateau_area_m2",
        "max_plateau_z_std_m",
        "min_aspect_ratio",
        "use_dominant_height_band",
        "slab_m",
        "height_percentile",
    ):
        if key not in cfg:
            cfg[key] = base.get(key)
    return cfg


def _build_gradient_search_base_mask(
    depth: np.ndarray,
    sobel_edges: np.ndarray,
    workspace_mask: np.ndarray | None,
    exclude_masks: list[np.ndarray] | None,
    *,
    edge_dilate_px: int,
    exclude_dilate_px: int,
) -> np.ndarray:
    """Workspace minus SAM3D parcels and Sobel edges — same basis as Stage-5 viz."""
    H, W = depth.shape
    if workspace_mask is not None:
        search = np.asarray(workspace_mask, dtype=bool).copy()
    else:
        search = np.ones((H, W), dtype=bool)
    search &= np.asarray(depth, dtype=np.float64) > 0

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

    edges = np.asarray(sobel_edges, dtype=np.uint8) > 0
    if edge_dilate_px > 0 and edges.any():
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * edge_dilate_px + 1, 2 * edge_dilate_px + 1),
        )
        edges = cv2.dilate(edges.astype(np.uint8), k, iterations=1) > 0
    search &= ~edges
    return search


def _enumerate_gradient_plateaus(
    plateau_mask: np.ndarray,
    depth: np.ndarray,
    plane: tuple[float, float, float, float],
    params: dict,
    *,
    id_offset: int = 0,
    store_masks: bool = False,
) -> tuple[list[GradientPlateau], dict, int]:
    """Connected components on `plateau_mask` → list of GradientPlateau."""
    min_plateau_area_px = int(params.get("min_plateau_area_px", 300))
    min_plateau_area_m2 = float(params.get("min_plateau_area_m2", 0.005))
    max_plateau_z_std = float(params.get("max_plateau_z_std_m", 0.015))
    min_aspect = float(params.get("min_aspect_ratio", 0.25))
    use_dominant = bool(params.get("use_dominant_height_band", True))
    slab_m = float(params.get("slab_m", 0.005))
    height_percentile = float(params.get("height_percentile", 95.0))

    depth = np.asarray(depth, dtype=np.float64)
    if not plateau_mask.any():
        return [], {}, 0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        plateau_mask.astype(np.uint8), connectivity=8,
    )

    plateaus: list[GradientPlateau] = []
    rej = {"area_px": 0, "aspect": 0, "area_m2": 0, "z_std": 0}
    next_gid = id_offset

    for li in range(1, n_labels):
        area_px = int(stats[li, cv2.CC_STAT_AREA])
        if area_px < min_plateau_area_px:
            rej["area_px"] += 1
            continue

        w_px = int(stats[li, cv2.CC_STAT_WIDTH])
        h_px = int(stats[li, cv2.CC_STAT_HEIGHT])
        aspect = min(w_px, h_px) / max(max(w_px, h_px), 1)
        if aspect < min_aspect:
            rej["aspect"] += 1
            continue

        ys, xs = np.where(labels == li)
        pts = _backproject_pixels(ys, xs, depth)
        heights = heights_above_plane(pts, plane)

        mean_z_cam = float(np.median(pts[:, 2]))
        area_m2 = area_px * (mean_z_cam / FX) * (mean_z_cam / FY)
        if area_m2 < min_plateau_area_m2:
            rej["area_m2"] += 1
            continue

        sigma_h = float(np.std(heights))
        if sigma_h > max_plateau_z_std:
            rej["z_std"] += 1
            continue

        if use_dominant:
            top = _dominant_height(heights, slab_m)
        else:
            top = float(np.percentile(heights, height_percentile))

        xy = pts[:, :2]
        footprint = (
            float(xy[:, 0].min()),
            float(xy[:, 1].min()),
            float(xy[:, 0].max()),
            float(xy[:, 1].max()),
        )
        sub_mask = None
        sub_pts = None
        if store_masks:
            H, W = depth.shape
            sub_mask = np.zeros((H, W), dtype=bool)
            sub_mask[ys, xs] = True
            sub_pts = pts.copy()

        plateaus.append(
            GradientPlateau(
                label=li,
                area_px=area_px,
                area_m2=area_m2,
                height_above_pallet=top,
                height_std_m=sigma_h,
                aspect_ratio=aspect,
                centroid_px=(float(xs.mean()), float(ys.mean())),
                global_id=next_gid,
                footprint_xy_cam=footprint,
                mask=sub_mask,
                points_3d=sub_pts,
            )
        )
        next_gid += 1

    return plateaus, rej, n_labels - 1


def _plateaus_from_cc_height_bands(
    ys: np.ndarray,
    xs: np.ndarray,
    depth: np.ndarray,
    plane: tuple[float, float, float, float],
    params: dict,
    *,
    min_band_pixels: int,
    min_band_fraction: float,
    store_masks: bool,
    id_start: int,
) -> list[GradientPlateau]:
    """Split one Sobel-connected region into several plateaus by height band.

    Without this, a large white floor patch in front of multiple boxes becomes
    a single plateau; per-box matching then only sees one shared surface.
    """
    H, W = depth.shape
    pts = _backproject_pixels(ys, xs, depth)
    heights = heights_above_plane(pts, plane)
    n_cc = heights.size
    if n_cc == 0:
        return []

    min_component_px = int(params.get("min_plateau_area_px", 80))
    slab_m = float(params.get("slab_m", 0.005))
    h_min, h_max = float(heights.min()), float(heights.max())
    if h_max - h_min < slab_m:
        hist = np.array([n_cc])
        edges_h = np.array([h_min, h_max + slab_m])
    else:
        n_bins = max(1, int(np.ceil((h_max - h_min) / slab_m)))
        hist, edges_h = np.histogram(heights, bins=n_bins)

    threshold = max(min_band_pixels, int(min_band_fraction * n_cc))
    out: list[GradientPlateau] = []
    next_gid = id_start

    for bi in range(len(hist)):
        if hist[bi] < threshold:
            continue
        lo, hi = float(edges_h[bi]), float(edges_h[bi + 1])
        in_band = (heights >= lo) & (heights <= hi)
        if not in_band.any():
            continue

        band_mask = np.zeros((H, W), dtype=np.uint8)
        band_mask[ys[in_band], xs[in_band]] = 1
        n_sl, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(
            band_mask, connectivity=8,
        )
        if n_sl <= 1:
            continue

        for sli in range(1, n_sl):
            area_px = int(sub_stats[sli, cv2.CC_STAT_AREA])
            if area_px < min_component_px:
                continue
            w_px = int(sub_stats[sli, cv2.CC_STAT_WIDTH])
            h_px = int(sub_stats[sli, cv2.CC_STAT_HEIGHT])
            aspect = min(w_px, h_px) / max(max(w_px, h_px), 1)
            if aspect < float(params.get("min_aspect_ratio", 0.05)):
                continue

            sub_ys, sub_xs = np.where(sub_labels == sli)
            sub_pts = _backproject_pixels(sub_ys, sub_xs, depth)
            sub_heights = heights_above_plane(sub_pts, plane)
            sigma_h = float(np.std(sub_heights))
            if sigma_h > float(params.get("max_plateau_z_std_m", 0.025)):
                continue

            mean_z_cam = float(np.median(sub_pts[:, 2]))
            area_m2 = area_px * (mean_z_cam / FX) * (mean_z_cam / FY)
            if area_m2 < float(params.get("min_plateau_area_m2", 0.001)):
                continue

            if bool(params.get("use_dominant_height_band", True)):
                top = _dominant_height(sub_heights, slab_m)
            else:
                top = float(np.percentile(
                    sub_heights, float(params.get("height_percentile", 95.0)),
                ))

            xy = sub_pts[:, :2]
            footprint = (
                float(xy[:, 0].min()), float(xy[:, 1].min()),
                float(xy[:, 0].max()), float(xy[:, 1].max()),
            )
            sub_mask = None
            sub_pts_store = None
            if store_masks:
                sub_mask = np.zeros((H, W), dtype=bool)
                sub_mask[sub_ys, sub_xs] = True
                sub_pts_store = sub_pts.copy()

            out.append(
                GradientPlateau(
                    label=sli,
                    area_px=area_px,
                    area_m2=area_m2,
                    height_above_pallet=top,
                    height_std_m=sigma_h,
                    aspect_ratio=aspect,
                    centroid_px=(float(sub_xs.mean()), float(sub_ys.mean())),
                    global_id=next_gid,
                    footprint_xy_cam=footprint,
                    mask=sub_mask,
                    points_3d=sub_pts_store,
                )
            )
            next_gid += 1

    return out


def detect_global_gradient_plateaus(
    depth: np.ndarray,
    sobel_edges: np.ndarray,
    workspace_mask: np.ndarray | None,
    exclude_masks: list[np.ndarray] | None,
    plane: tuple[float, float, float, float],
    config: dict,
) -> list[GradientPlateau]:
    """Once per scene: all Sobel-separated plateaus in the workspace.

    Uses the same edge map as the gradient visualisation (Stage 5/6).
    SAM3D parcel masks are excluded so parcel tops are not counted as
    neighbour surfaces.  Per-parcel bottom inference then queries this
    catalogue instead of re-segmenting only a narrow ring.
    """
    cfg = config.get("global_gradient_plateaus", {})
    if not cfg.get("enabled", True):
        return []

    edge_dilate = int(cfg.get("edge_dilate_px", config.get("gradient_neighbor", {}).get("edge_dilate_px", 1)))
    exclude_dilate = int(cfg.get("exclude_dilate_px", 4))
    params = _plateau_filter_cfg(config, "global_gradient_plateaus")

    min_cc_px = int(cfg.get("min_component_px", 80))
    min_band_pixels = int(cfg.get("min_band_pixels", 40))
    min_band_fraction = float(cfg.get("min_band_fraction", 0.02))
    split_bands = bool(cfg.get("split_height_bands", True))

    search = _build_gradient_search_base_mask(
        depth, sobel_edges, workspace_mask, exclude_masks,
        edge_dilate_px=edge_dilate, exclude_dilate_px=exclude_dilate,
    )
    depth_arr = np.asarray(depth, dtype=np.float64)
    H, W = depth_arr.shape

    if not split_bands:
        plateaus, _, _ = _enumerate_gradient_plateaus(
            search, depth_arr, plane, params, id_offset=0, store_masks=True,
        )
        return plateaus

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        search.astype(np.uint8), connectivity=8,
    )
    plateaus: list[GradientPlateau] = []
    next_id = 0
    for li in range(1, n_labels):
        if int(stats[li, cv2.CC_STAT_AREA]) < min_cc_px:
            continue
        ys, xs = np.where(labels == li)
        bands = _plateaus_from_cc_height_bands(
            ys, xs, depth_arr, plane, params,
            min_band_pixels=min_band_pixels,
            min_band_fraction=min_band_fraction,
            store_masks=True,
            id_start=next_id,
        )
        plateaus.extend(bands)
        next_id += len(bands)

    return plateaus


def _target_footprint_xy(target: CandidateOut) -> tuple[float, float, float, float]:
    if len(target.points_3d) > 0:
        xy = np.asarray(target.points_3d, dtype=np.float64)[:, :2]
        return (
            float(xy[:, 0].min()),
            float(xy[:, 1].min()),
            float(xy[:, 0].max()),
            float(xy[:, 1].max()),
        )
    return (0.0, 0.0, 0.0, 0.0)


def _xy_overlap_min_area(
    fp_a: tuple[float, float, float, float],
    fp_b: tuple[float, float, float, float],
) -> float:
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
class GlobalGradientMatchResult:
    z_highest_neighbor: float | None
    chosen_global_id: int | None
    matching_plateaus: list[GradientPlateau]
    rejection_counts: dict | None = None


def _build_target_search_mask(
    target: CandidateOut,
    depth: np.ndarray,
    search_pad_m: float,
) -> np.ndarray:
    """Dilated 2D search region: SAM mask + pad so front/side neighbour planes count."""
    target_mask = np.asarray(target.mask_2d, dtype=np.uint8) > 0
    H, W = depth.shape
    if target_mask.sum() == 0 or search_pad_m <= 0:
        return target_mask

    td = depth[target_mask]
    valid = td[td > 0]
    mean_z = float(np.median(valid)) if valid.size > 0 else 1.0
    pad_px = max(5, int(round(search_pad_m * FX / max(mean_z, 0.1))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * pad_px + 1, 2 * pad_px + 1),
    )
    dilated = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1) > 0
    return dilated


def _expanded_footprint(
    fp: tuple[float, float, float, float],
    pad_m: float,
) -> tuple[float, float, float, float]:
    return (fp[0] - pad_m, fp[1] - pad_m, fp[2] + pad_m, fp[3] + pad_m)


def find_neighbor_from_gradient_catalog(
    target: CandidateOut,
    catalog: list[GradientPlateau],
    z_visible_min: float,
    config: dict,
    depth: np.ndarray | None = None,
) -> GlobalGradientMatchResult:
    """Query the global gradient catalogue for the best supporting surface.

    Spatial match (any one is enough):
      - pixel overlap between plateau mask and dilated target mask
      - XY footprint overlap with padded target footprint
      - plateau centroid within max_centroid_dist_m of target centre

    Height: plateau top must be strictly below z_visible_min - tolerance.
    """
    cfg = config.get("global_gradient_plateaus", {})
    tol = float(config.get("tolerance_m", 0.008))
    min_overlap = float(cfg.get("min_overlap", 0.02))
    max_centroid_dist_m = float(cfg.get("max_centroid_dist_m", 0.45))
    search_pad_m = float(cfg.get("search_pad_m", 0.25))

    if not catalog:
        return GlobalGradientMatchResult(None, None, [], {"no_catalog": 1})

    tgt_fp = _target_footprint_xy(target)
    tgt_fp_pad = _expanded_footprint(tgt_fp, search_pad_m)
    if len(target.points_3d) > 0:
        tgt_center = np.asarray(target.points_3d, dtype=np.float64)[:, :2].mean(axis=0)
    else:
        tgt_center = np.array([
            0.5 * (tgt_fp[0] + tgt_fp[2]),
            0.5 * (tgt_fp[1] + tgt_fp[3]),
        ])

    search_mask = None
    if depth is not None:
        search_mask = _build_target_search_mask(target, depth, search_pad_m)

    rej = {"too_high": 0, "no_overlap": 0}
    matching: list[GradientPlateau] = []

    for p in catalog:
        if p.footprint_xy_cam is None:
            continue
        if p.height_above_pallet >= z_visible_min - tol:
            rej["too_high"] += 1
            continue

        spatial_ok = False

        if search_mask is not None and p.mask is not None and p.mask.any():
            if (p.mask & search_mask).any():
                spatial_ok = True

        if not spatial_ok:
            overlap = _xy_overlap_min_area(tgt_fp_pad, p.footprint_xy_cam)
            if overlap >= min_overlap:
                spatial_ok = True

        if not spatial_ok:
            cx = 0.5 * (p.footprint_xy_cam[0] + p.footprint_xy_cam[2])
            cy = 0.5 * (p.footprint_xy_cam[1] + p.footprint_xy_cam[3])
            dist = float(np.hypot(cx - tgt_center[0], cy - tgt_center[1]))
            if dist <= max_centroid_dist_m:
                spatial_ok = True

        if not spatial_ok:
            rej["no_overlap"] += 1
            continue

        matching.append(p)

    if not matching:
        return GlobalGradientMatchResult(None, None, [], rej)

    best = max(matching, key=lambda p: p.height_above_pallet)
    return GlobalGradientMatchResult(
        z_highest_neighbor=best.height_above_pallet,
        chosen_global_id=best.global_id,
        matching_plateaus=matching,
        rejection_counts=rej,
    )


def find_neighbor_via_gradient(
    target: CandidateOut,
    depth: np.ndarray,
    sobel_edges: np.ndarray,
    workspace_mask: np.ndarray | None,
    plane: tuple[float, float, float, float],
    z_visible_min: float,
    config: dict,
    obb_extent_xy_m: float | None = None,
) -> GradientNeighborResult:
    """
    Pure-Sobel neighbourhood analysis (no DINO, no Stage-5 matches).

    1. Define the neighbourhood as a ring around the target mask. The
       effective radius is adaptive:
            r_m = max(neighbor_radius_m, adaptive_radius_factor * obb_extent_xy_m)
       so large parcels search a proportionally larger ring.
       Pixel radius is derived from `r_m` via the parcel's mean depth
       (fallback: `neighbor_radius_px`).
    2. Inside the ring, a "plateau" = connected component of pixels that
       are NOT marked as a Sobel/Canny edge AND have a valid depth value.
    3. For each plateau (area >= `min_plateau_area_px`):
         - Backproject all pixels, compute height_above_pallet,
         - Use a robust upper estimate (`height_percentile`) as the plateau's
           top height (the user's 'höchster Punkt' on the plateau).
    4. Return the highest plateau whose top is strictly below
       z_visible_min - tolerance.
    """
    cfg = config.get("gradient_neighbor", {})
    radius_px_fallback = int(cfg.get("neighbor_radius_px", 60))
    radius_m_min = cfg.get("neighbor_radius_m", None)
    adaptive_factor = float(cfg.get("adaptive_radius_factor", 1.0))
    max_radius_m = float(cfg.get("max_radius_m", 1.0))
    edge_dilate_px = int(cfg.get("edge_dilate_px", 1))
    params = _plateau_filter_cfg(config, "gradient_neighbor")
    tol = float(config.get("tolerance_m", 0.008))

    target_mask = np.asarray(target.mask_2d, dtype=np.uint8) > 0
    edges = np.asarray(sobel_edges, dtype=np.uint8) > 0
    depth = np.asarray(depth, dtype=np.float64)

    if target_mask.sum() == 0:
        return GradientNeighborResult(None, None, [], 0, 0, {}, float("nan"), 0)

    if radius_m_min is not None:
        target_depth = depth[target_mask]
        mean_depth = float(np.median(target_depth[target_depth > 0])) if (target_depth > 0).any() else 1.0

        base = float(radius_m_min)
        if obb_extent_xy_m is not None and adaptive_factor > 0:
            adaptive = float(obb_extent_xy_m) * adaptive_factor
            effective_radius_m = max(base, adaptive)
        else:
            effective_radius_m = base
        effective_radius_m = min(effective_radius_m, max_radius_m)

        radius_px = max(10, int(round(effective_radius_m * FX / max(mean_depth, 0.1))))
    else:
        radius_px = radius_px_fallback
        effective_radius_m = float("nan")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
    )
    dilated = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1) > 0
    ring = dilated & ~target_mask
    if workspace_mask is not None:
        ring &= np.asarray(workspace_mask, dtype=bool)
    ring &= depth > 0

    if edge_dilate_px > 0:
        ek = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * edge_dilate_px + 1, 2 * edge_dilate_px + 1)
        )
        edges = cv2.dilate(edges.astype(np.uint8), ek, iterations=1) > 0

    plateau_mask = ring & ~edges
    n_ring = int(ring.sum())
    if not plateau_mask.any():
        return GradientNeighborResult(
            None, None, [], n_ring, 0, {}, effective_radius_m, radius_px,
        )

    plateaus, rej, n_components = _enumerate_gradient_plateaus(
        plateau_mask, depth, plane, params,
    )

    qualifying = [p for p in plateaus if p.height_above_pallet < z_visible_min - tol]
    if not qualifying:
        return GradientNeighborResult(
            None, None, plateaus, n_ring, n_components, rej,
            effective_radius_m, radius_px,
        )

    best = max(qualifying, key=lambda p: p.height_above_pallet)
    return GradientNeighborResult(
        z_highest_neighbor=best.height_above_pallet,
        chosen_label=best.label,
        plateaus=plateaus,
        n_ring_pixels=n_ring,
        n_components_total=n_components,
        rejection_counts=rej,
        effective_radius_m=effective_radius_m,
        radius_px=radius_px,
    )


# ---------------------------------------------------------------------------
# Depth-histogram plateau detection (Sobel-independent fallback)
# ---------------------------------------------------------------------------

@dataclass
class DepthBand:
    """A horizontal slab of points in the neighbourhood ring (identified
    purely by depth histogramming, no Sobel required)."""
    band_low: float
    band_high: float
    height_median: float
    area_px: int
    area_m2: float
    aspect_ratio: float
    centroid_px: tuple[float, float]


@dataclass
class DepthHistogramResult:
    z_highest_neighbor: float | None
    bands: list[DepthBand]
    n_ring_pixels: int
    effective_radius_m: float = float("nan")
    radius_px: int = 0
    rejection_counts: dict | None = None


def find_neighbor_via_depth_histogram(
    target: CandidateOut,
    depth: np.ndarray,
    workspace_mask: np.ndarray | None,
    plane: tuple[float, float, float, float],
    z_visible_min: float,
    config: dict,
    obb_extent_xy_m: float | None = None,
) -> DepthHistogramResult:
    """Sobel-free plateau detection.

    Procedure:
      1. Build the same adaptive ring as the gradient detector.
      2. Compute height-above-pallet for every valid pixel in the ring.
      3. Bin the heights into `slab_m` slabs and accept every slab whose
         pixel count exceeds `min_band_pixels` OR `min_band_fraction` of
         the ring.
      4. Per accepted band, run 2D connected components on the in-band
         pixels and keep the largest component that passes
         (min_component_area_m2, min_component_aspect).
      5. Band's "top" = median height inside that band. Return the
         highest band whose top is strictly below z_visible_min - tol.

    Works when Sobel is too noisy / misses thin edges between adjacent
    cardboard parcels at the same colour: the depth jump itself is enough
    to separate them in the height histogram.
    """
    cfg = config.get("depth_histogram", {})
    if not cfg.get("enabled", True):
        return DepthHistogramResult(None, [], 0)

    radius_min_m = cfg.get("neighbor_radius_m", config.get("gradient_neighbor", {}).get("neighbor_radius_m", 0.30))
    adaptive_factor = float(cfg.get("adaptive_radius_factor",
                                    config.get("gradient_neighbor", {}).get("adaptive_radius_factor", 1.0)))
    max_radius_m = float(cfg.get("max_radius_m",
                                 config.get("gradient_neighbor", {}).get("max_radius_m", 1.0)))
    fallback_radius_px = int(cfg.get("neighbor_radius_px", 80))
    slab_m = float(cfg.get("slab_m", 0.005))
    min_band_pixels = int(cfg.get("min_band_pixels", 200))
    min_band_fraction = float(cfg.get("min_band_fraction", 0.02))
    min_area_m2 = float(cfg.get("min_component_area_m2", 0.003))
    min_aspect = float(cfg.get("min_component_aspect", 0.20))
    tol = float(config.get("tolerance_m", 0.008))

    target_mask = np.asarray(target.mask_2d, dtype=np.uint8) > 0
    depth = np.asarray(depth, dtype=np.float64)
    if target_mask.sum() == 0:
        return DepthHistogramResult(None, [], 0)

    ring, effective_radius_m, radius_px = _build_neighborhood_ring(
        target_mask, depth, workspace_mask,
        radius_min_m, adaptive_factor, max_radius_m,
        obb_extent_xy_m, fallback_radius_px,
    )
    n_ring = int(ring.sum())
    if n_ring == 0:
        return DepthHistogramResult(None, [], 0, effective_radius_m, radius_px, {})

    ys, xs = np.where(ring)
    pts = _backproject_pixels(ys, xs, depth)
    heights = heights_above_plane(pts, plane)

    h_min, h_max = float(heights.min()), float(heights.max())
    if h_max - h_min < slab_m:
        n_bins = 1
    else:
        n_bins = max(1, int(np.ceil((h_max - h_min) / slab_m)))
    hist, edges = np.histogram(heights, bins=n_bins)

    pixel_threshold = max(min_band_pixels, int(min_band_fraction * n_ring))

    bands: list[DepthBand] = []
    rej = {"low_pixels": 0, "no_component": 0, "area_m2": 0, "aspect": 0}

    for bi in range(len(hist)):
        if hist[bi] < pixel_threshold:
            rej["low_pixels"] += 1
            continue
        lo, hi = float(edges[bi]), float(edges[bi + 1])
        in_band = (heights >= lo) & (heights <= hi)
        if not in_band.any():
            rej["low_pixels"] += 1
            continue

        band_mask = np.zeros_like(ring, dtype=np.uint8)
        band_mask[ys[in_band], xs[in_band]] = 1

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            band_mask, connectivity=8,
        )
        if n_labels <= 1:
            rej["no_component"] += 1
            continue

        best_li = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        area_px = int(stats[best_li, cv2.CC_STAT_AREA])
        w_px = int(stats[best_li, cv2.CC_STAT_WIDTH])
        h_px = int(stats[best_li, cv2.CC_STAT_HEIGHT])
        aspect = min(w_px, h_px) / max(max(w_px, h_px), 1)
        if aspect < min_aspect:
            rej["aspect"] += 1
            continue

        cys, cxs = np.where(labels == best_li)
        comp_pts = _backproject_pixels(cys, cxs, depth)
        mean_zcam = float(np.median(comp_pts[:, 2]))
        area_m2 = area_px * (mean_zcam / FX) * (mean_zcam / FY)
        if area_m2 < min_area_m2:
            rej["area_m2"] += 1
            continue

        comp_heights = heights_above_plane(comp_pts, plane)
        bands.append(DepthBand(
            band_low=lo,
            band_high=hi,
            height_median=float(np.median(comp_heights)),
            area_px=area_px,
            area_m2=area_m2,
            aspect_ratio=aspect,
            centroid_px=(float(cxs.mean()), float(cys.mean())),
        ))

    qualifying = [b for b in bands if b.height_median < z_visible_min - tol]
    if not qualifying:
        return DepthHistogramResult(
            None, bands, n_ring, effective_radius_m, radius_px, rej,
        )

    best = max(qualifying, key=lambda b: b.height_median)
    return DepthHistogramResult(
        z_highest_neighbor=best.height_median,
        bands=bands,
        n_ring_pixels=n_ring,
        effective_radius_m=effective_radius_m,
        radius_px=radius_px,
        rejection_counts=rej,
    )


def footprint_iou(
    fp_a: tuple[float, float, float, float],
    fp_b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = fp_a
    bx0, by0, bx1, by1 = fp_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 1e-12 else 0.0
