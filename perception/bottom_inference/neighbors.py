"""Neighbour finding and solid-surface detection for bottom-plane inference."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from perception.candidate import CandidateOut
from perception.geometry.plane import heights_above_plane, project_to_plane_xy

# Default intrinsics – kept in sync with perception.adapter.FX/FY.
FX = FY = 437.04


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
    """Detected-candidate neighbours (only used as a sanity hint)."""
    lateral_radius_m = float(config["lateral_radius_m"])
    height_tolerance = float(config["height_tolerance"])

    r_neighbor = lateral_radius_m + 0.5 * target.obb_extent_xy
    ids: list[str] = []
    tops: list[float] = []
    dists: list[float] = []

    others = [g for cid, g in all_geom.items() if cid != target.candidate_id]
    if not others:
        return NeighborInfo([], [], [], None, None, None, None)

    centers = np.array([g.center_xy for g in others], dtype=np.float64)
    tree = cKDTree(centers)
    idxs = tree.query_ball_point(target.center_xy, r_neighbor)

    for i in idxs:
        g = others[i]
        if g.top_surface_height >= target.top_surface_height - height_tolerance:
            continue
        dist = float(np.linalg.norm(g.center_xy - target.center_xy))
        ids.append(g.candidate_id)
        tops.append(g.top_surface_height)
        dists.append(dist)

    order = np.argsort(ids)
    ids = [ids[i] for i in order]
    tops = [tops[i] for i in order]
    dists = [dists[i] for i in order]

    if not ids:
        return NeighborInfo([], [], [], None, None, None, None)

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
    )


@dataclass
class GradientPlateau:
    """A flat region in the neighbourhood of a parcel, separated from
    other regions by Sobel edges."""
    label: int
    area_px: int
    height_above_pallet: float       # robust top-height of the plateau
    centroid_px: tuple[float, float]


@dataclass
class GradientNeighborResult:
    z_highest_neighbor: float | None
    chosen_label: int | None
    plateaus: list[GradientPlateau]
    n_ring_pixels: int


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


def find_neighbor_via_gradient(
    target: CandidateOut,
    depth: np.ndarray,
    sobel_edges: np.ndarray,
    workspace_mask: np.ndarray | None,
    plane: tuple[float, float, float, float],
    z_visible_min: float,
    config: dict,
) -> GradientNeighborResult:
    """
    Pure-Sobel neighbourhood analysis (no DINO, no Stage-5 matches).

    1. Define the neighbourhood as a ring around the target mask
       (dilate by `neighbor_radius_m` converted to pixels via the
       parcel's mean depth, fallback to `neighbor_radius_px`).
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
    radius_m = cfg.get("neighbor_radius_m", None)
    min_plateau_area_px = int(cfg.get("min_plateau_area_px", 400))
    edge_dilate_px = int(cfg.get("edge_dilate_px", 1))
    height_percentile = float(cfg.get("height_percentile", 95.0))
    tol = float(config.get("tolerance_m", 0.008))

    target_mask = np.asarray(target.mask_2d, dtype=np.uint8) > 0
    edges = np.asarray(sobel_edges, dtype=np.uint8) > 0
    depth = np.asarray(depth, dtype=np.float64)

    if target_mask.sum() == 0:
        return GradientNeighborResult(None, None, [], 0)

    # Radius in Metern -> Pixel via mittlerer Tiefe der Target-Maske.
    if radius_m is not None:
        target_depth = depth[target_mask]
        mean_depth = float(np.median(target_depth[target_depth > 0])) if (target_depth > 0).any() else 1.0
        radius_px = max(10, int(round(float(radius_m) * FX / max(mean_depth, 0.1))))
    else:
        radius_px = radius_px_fallback

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
        return GradientNeighborResult(None, None, [], n_ring)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        plateau_mask.astype(np.uint8), connectivity=8
    )

    plateaus: list[GradientPlateau] = []
    for li in range(1, n_labels):
        area = int(stats[li, cv2.CC_STAT_AREA])
        if area < min_plateau_area_px:
            continue
        ys, xs = np.where(labels == li)
        pts = _backproject_pixels(ys, xs, depth)
        heights = heights_above_plane(pts, plane)
        top = float(np.percentile(heights, height_percentile))
        cx_px = float(xs.mean())
        cy_px = float(ys.mean())
        plateaus.append(
            GradientPlateau(
                label=li,
                area_px=area,
                height_above_pallet=top,
                centroid_px=(cx_px, cy_px),
            )
        )

    qualifying = [p for p in plateaus if p.height_above_pallet < z_visible_min - tol]
    if not qualifying:
        return GradientNeighborResult(None, None, plateaus, n_ring)

    best = max(qualifying, key=lambda p: p.height_above_pallet)
    return GradientNeighborResult(
        z_highest_neighbor=best.height_above_pallet,
        chosen_label=best.label,
        plateaus=plateaus,
        n_ring_pixels=n_ring,
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
