"""The three verification stages (BBox validity, suctionability, clearance).

Each stage returns a StageResult with continuous-margin CheckRecords plus
outputs consumed by later stages. Stages never short-circuit internally; the
cascade vs full decision is made by the orchestrator in verify.py.
"""

from __future__ import annotations

import numpy as np

from perception.geometry.plane import heights_above_plane, project_to_plane_xy
from verification.geometry import (
    Intrinsics,
    _gripper_rot,
    angle_between,
    gather_gripper_points,
    robust_plane_fit,
)
from verification.config import resolve_corridor_height, resolve_gripper
from verification.types import CheckRecord, StageResult

_LARGE_MARGIN = 1.0


def _cluster_by_gap(values: np.ndarray, gap: float) -> list[np.ndarray]:
    """Split sorted 1D values into groups separated by a gap > `gap`."""
    if values.size == 0:
        return []
    vs = np.sort(values)
    splits = np.where(np.diff(vs) > gap)[0]
    return np.split(vs, splits + 1)


def _seam_span(sub_depth: np.ndarray, step_m: float) -> float:
    """Largest fraction of a bbox row/column line crossed by a depth step.

    Returns max over column boundaries (vertical seam) and row boundaries
    (horizontal seam) of the fraction of valid neighbour pairs whose depth jump
    exceeds `step_m`. ~1.0 means a step spans (almost) the whole bbox.
    """
    d = sub_depth.astype(np.float64)
    d[d <= 0] = np.nan
    spans = [0.0]

    if d.shape[1] >= 2:
        col_diff = np.abs(d[:, 1:] - d[:, :-1])  # (H, W-1)
        with np.errstate(invalid="ignore"):
            exceed = col_diff > step_m
        valid = ~np.isnan(col_diff)
        denom = valid.sum(axis=0).astype(np.float64)
        denom[denom == 0] = np.nan
        frac = exceed.sum(axis=0) / denom
        spans.append(float(np.nanmax(frac)) if np.any(~np.isnan(frac)) else 0.0)

    if d.shape[0] >= 2:
        row_diff = np.abs(d[1:, :] - d[:-1, :])  # (H-1, W)
        with np.errstate(invalid="ignore"):
            exceed = row_diff > step_m
        valid = ~np.isnan(row_diff)
        denom = valid.sum(axis=1).astype(np.float64)
        denom[denom == 0] = np.nan
        frac = exceed.sum(axis=1) / denom
        spans.append(float(np.nanmax(frac)) if np.any(~np.isnan(frac)) else 0.0)

    return max(spans)


def run_stage1(
    p_bbox: np.ndarray,
    n_valid: int,
    n_bbox_px: int,
    sub_depth: np.ndarray,
    plane: tuple[float, float, float, float],
    cfg: dict,
) -> StageResult:
    """Stage 1 - is the bounding box a valid single-object premise?"""
    s = cfg["stage1"]
    tau_valid = float(s["tau_valid"])
    gap = float(s["plateau_gap_m"])
    top_min_frac = float(s["top_cluster_min_fraction"])
    step_m = float(s["depth_edge_step_m"])
    span_ratio = float(s["depth_edge_span_ratio"])

    checks: list[CheckRecord] = []

    # --- Check 1a: data quality + top face determinable ---
    valid_ratio = (n_valid / n_bbox_px) if n_bbox_px > 0 else 0.0
    checks.append(
        CheckRecord(
            name="valid_ratio",
            stage=1,
            raw_value=valid_ratio,
            threshold=tau_valid,
            margin=valid_ratio - tau_valid,
            passed=valid_ratio >= tau_valid,
            detail={"n_valid": int(n_valid), "n_bbox_px": int(n_bbox_px)},
        )
    )

    heights = heights_above_plane(p_bbox, plane) if len(p_bbox) else np.zeros(0)
    z_top = None
    top_mask = None
    top_frac = 0.0
    groups: list[np.ndarray] = []
    if heights.size > 0:
        groups = _cluster_by_gap(heights, gap)
        total = float(heights.size)
        group_fracs = [(g, len(g) / total) for g in groups]
        # Top face = highest group; representative height = 95th percentile.
        top_group = max(groups, key=lambda g: g.mean())
        top_frac = len(top_group) / total
        z_top = float(np.percentile(top_group, 95))
        top_mask = heights >= (top_group.min() - 1e-9)

    checks.append(
        CheckRecord(
            name="top_cluster",
            stage=1,
            raw_value=top_frac,
            threshold=top_min_frac,
            margin=top_frac - top_min_frac,
            passed=(z_top is not None) and (top_frac >= top_min_frac),
            detail={"z_top_m": z_top, "n_groups": len(groups)},
        )
    )

    # --- Check 1b: single object (no second plateau, no through-seam) ---
    dominant = [g for g in groups if (len(g) / max(len(heights), 1)) >= top_min_frac]
    if len(dominant) >= 2:
        dominant_sorted = sorted(dominant, key=lambda g: g.mean())
        # Gap between the two highest dominant plateaus.
        g_hi = dominant_sorted[-1]
        g_lo = dominant_sorted[-2]
        plateau_gap = float(g_hi.min() - g_lo.max())
        single_passed = plateau_gap <= gap
        single_margin = gap - plateau_gap
        single_raw = plateau_gap
    else:
        single_passed = True
        single_margin = _LARGE_MARGIN
        single_raw = 0.0
    checks.append(
        CheckRecord(
            name="single_object",
            stage=1,
            raw_value=single_raw,
            threshold=gap,
            margin=single_margin,
            passed=single_passed,
            detail={"n_dominant_plateaus": len(dominant)},
        )
    )

    seam_span = _seam_span(sub_depth, step_m) if sub_depth.size else 0.0
    checks.append(
        CheckRecord(
            name="no_seam",
            stage=1,
            raw_value=seam_span,
            threshold=span_ratio,
            margin=span_ratio - seam_span,
            passed=seam_span < span_ratio,
            detail={"depth_step_m": step_m},
        )
    )

    passed = all(c.passed for c in checks)
    return StageResult(
        stage=1,
        name="bbox_valid",
        passed=passed,
        checks=checks,
        outputs={"z_top": z_top, "top_mask": top_mask, "valid_ratio": valid_ratio},
    )


def _normal_scatter(
    points: np.ndarray,
    rel_xy: np.ndarray,
    mean_normal: np.ndarray,
    n_cells: int = 2,
) -> float:
    """Dispersion of local plane normals over a coarse grid of the gripper window.

    Returns the max angle (deg, normalised to [0,1] by /90) between any cell's
    local normal and the mean normal. Captures rugged / folded surfaces.
    """
    if len(points) < 4 * n_cells:
        return 0.0
    xs, ys = rel_xy[:, 0], rel_xy[:, 1]
    r = float(max(np.abs(xs).max(), np.abs(ys).max(), 1e-6))
    edges = np.linspace(-r, r, n_cells + 1)
    angles = []
    for i in range(n_cells):
        for j in range(n_cells):
            cell = (
                (xs >= edges[i]) & (xs < edges[i + 1])
                & (ys >= edges[j]) & (ys < edges[j + 1])
            )
            if cell.sum() < 6:
                continue
            pts = points[cell]
            centroid = pts.mean(axis=0)
            cov = np.cov((pts - centroid).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            local_n = eigvecs[:, 0]
            angles.append(angle_between(local_n, mean_normal))
    if not angles:
        return 0.0
    return float(max(angles) / 90.0)


def _dilate4(grid: np.ndarray) -> np.ndarray:
    """4-neighbour binary dilation (no SciPy dependency)."""
    out = grid.copy()
    out[1:, :] |= grid[:-1, :]
    out[:-1, :] |= grid[1:, :]
    out[:, 1:] |= grid[:, :-1]
    out[:, :-1] |= grid[:, 1:]
    return out


def _erode4(grid: np.ndarray) -> np.ndarray:
    """4-neighbour binary erosion (interior cells only; border erodes)."""
    out = grid.copy()
    out[1:, :] &= grid[:-1, :]
    out[:-1, :] &= grid[1:, :]
    out[:, 1:] &= grid[:, :-1]
    out[:, :-1] &= grid[:, 1:]
    return out


def _binary_close(grid: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological closing: fills small sampling gaps without growing edges."""
    out = grid
    for _ in range(max(0, int(iterations))):
        out = _dilate4(out)
    for _ in range(max(0, int(iterations))):
        out = _erode4(out)
    return out


def _gripper_footprint_and_edge(
    rel_xy: np.ndarray,
    inlier_mask: np.ndarray,
    half_w: float,
    half_l: float,
    raster_m: float,
    close_iter: int = 2,
) -> tuple[float, float]:
    """Contiguous flat coverage of the rectangular gripper footprint.

    Robust to sub-cell sampling gaps: cells hit by a plane inlier are marked
    flat, then morphological closing bridges holes caused by the raster being
    finer than the depth sampling (≈ depth/fx). Coverage is the fraction of the
    footprint that is flat AND connected to the centre; edge clearance is the
    distance from the centre to the nearest genuine non-flat (edge/hole) cell.

    Returns (coverage_fraction, edge_clearance_m).
    """
    if rel_xy.shape[0] == 0:
        return 0.0, 0.0
    nx = max(1, int(np.ceil(2 * half_w / raster_m)))
    ny = max(1, int(np.ceil(2 * half_l / raster_m)))

    def to_cell(xy: np.ndarray) -> np.ndarray:
        ix = np.floor((xy[:, 0] + half_w) / raster_m).astype(int)
        iy = np.floor((xy[:, 1] + half_l) / raster_m).astype(int)
        return np.stack(
            [np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)],
            axis=1,
        )

    center_idx = (
        min(int(np.floor(half_w / raster_m)), nx - 1),
        min(int(np.floor(half_l / raster_m)), ny - 1),
    )

    flat = np.zeros((nx, ny), dtype=bool)
    inl = rel_xy[inlier_mask]
    if len(inl):
        cells = to_cell(inl)
        flat[cells[:, 0], cells[:, 1]] = True

    # Bridge sampling gaps (empty interior cells) without inflating real edges.
    flat = _binary_close(flat, iterations=close_iter)

    if not flat[center_idx] and len(inl):
        if float(np.linalg.norm(inl, axis=1).min()) <= 1.5 * raster_m:
            flat[center_idx] = True

    cx = (np.arange(nx) + 0.5) * raster_m - half_w
    cy = (np.arange(ny) + 0.5) * raster_m - half_l
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    required = (np.abs(gx) <= half_w) & (np.abs(gy) <= half_l)
    dist_to_center = np.sqrt(gx**2 + gy**2)
    n_required = int(required.sum())
    if n_required == 0:
        return 0.0, 0.0

    component = np.zeros((nx, ny), dtype=bool)
    if flat[center_idx]:
        stack = [center_idx]
        component[center_idx] = True
        while stack:
            ci, cj = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < nx and 0 <= nj < ny and flat[ni, nj] and not component[ni, nj]:
                    component[ni, nj] = True
                    stack.append((ni, nj))

    coverage = float((component & required).sum()) / float(n_required)

    bad = required & (~component)
    if bad.any():
        edge_clearance = float(dist_to_center[bad].min())
    else:
        edge_clearance = float(min(half_w, half_l))
    return coverage, edge_clearance


def run_stage2(
    p_target: np.ndarray,
    p_g: np.ndarray,
    plane: tuple[float, float, float, float],
    approach_axis: np.ndarray,
    cfg: dict,
    long_dir_xy: np.ndarray | None = None,
) -> StageResult:
    """Stage 2 - is the grasp point suctionable (planar, aligned, sealed)?

    The grasp position is the centre of the configured rectangular gripper
    footprint. ``p_target`` must contain only points on the selected parcel.
    ``long_dir_xy`` (pallet-plane unit vector) orients the gripper's long side
    along the parcel's longer side; None falls back to the u axis.
    """
    s = cfg["stage2"]
    grip = resolve_gripper(cfg)
    # Long side of the gripper aligns with the parcel long axis.
    half_long = max(grip.half_w_m, grip.half_l_m)
    half_short = min(grip.half_w_m, grip.half_l_m)
    rmse_max = float(s["plane_rmse_max_m"])
    angle_max = float(s["normal_angle_max_deg"])
    scatter_max = float(s["normal_scatter_max"])
    min_area_ratio = float(s["min_area_ratio"])
    raster_m = float(s["raster_m"])
    edge_min = s["edge_clearance_min_m"]
    edge_min = float(edge_min) if edge_min is not None else half_short
    rf = s["robust_fit"]

    p_grip, rel_xy = gather_gripper_points(
        p_target, p_g, half_long, half_short, plane, long_dir_xy
    )

    checks: list[CheckRecord] = []
    plane_point = p_g
    plane_normal = approach_axis

    if len(p_grip) < int(rf["min_points"]):
        checks.append(
            CheckRecord(
                name="planarity", stage=2, raw_value=float("inf"),
                threshold=rmse_max, margin=-_LARGE_MARGIN, passed=False,
                detail={"n_gripper_points": int(len(p_grip))},
            )
        )
        checks.append(
            CheckRecord(
                name="normal_angle", stage=2, raw_value=90.0,
                threshold=angle_max, margin=-_LARGE_MARGIN, passed=False,
            )
        )
        checks.append(
            CheckRecord(
                name="normal_scatter", stage=2, raw_value=1.0,
                threshold=scatter_max, margin=-_LARGE_MARGIN, passed=False,
            )
        )
        checks.append(
            CheckRecord(
                name="suction_area", stage=2, raw_value=0.0,
                threshold=min_area_ratio, margin=-_LARGE_MARGIN, passed=False,
            )
        )
        checks.append(
            CheckRecord(
                name="edge_clearance", stage=2, raw_value=0.0,
                threshold=edge_min, margin=-edge_min, passed=False,
            )
        )
        return StageResult(
            stage=2, name="grasp_suctionable", passed=False, checks=checks,
            outputs={
                "plane_point": plane_point,
                "plane_normal": plane_normal,
                "n_gripper_points": int(len(p_grip)),
                "gripper_width_m": grip.width_m,
                "gripper_length_m": grip.length_m,
            },
        )

    plane_point, plane_normal, rmse, inlier_mask = robust_plane_fit(
        p_grip,
        max_iter=int(rf["max_iter"]),
        mad_scale=float(rf["mad_scale"]),
        min_points=int(rf["min_points"]),
    )

    # 2a planarity
    checks.append(
        CheckRecord(
            name="planarity", stage=2, raw_value=rmse, threshold=rmse_max,
            margin=rmse_max - rmse, passed=rmse <= rmse_max,
            detail={"n_inliers": int(inlier_mask.sum())},
        )
    )

    # 2b normal consistency
    theta = angle_between(plane_normal, approach_axis)
    checks.append(
        CheckRecord(
            name="normal_angle", stage=2, raw_value=theta, threshold=angle_max,
            margin=angle_max - theta, passed=theta <= angle_max,
        )
    )
    scatter = _normal_scatter(p_grip, rel_xy, plane_normal)
    checks.append(
        CheckRecord(
            name="normal_scatter", stage=2, raw_value=scatter,
            threshold=scatter_max, margin=scatter_max - scatter,
            passed=scatter <= scatter_max,
        )
    )

    # 2c min suction area + edge clearance
    coverage, edge_clearance = _gripper_footprint_and_edge(
        rel_xy, inlier_mask, half_long, half_short, raster_m
    )
    checks.append(
        CheckRecord(
            name="suction_area", stage=2, raw_value=coverage,
            threshold=min_area_ratio, margin=coverage - min_area_ratio,
            passed=coverage >= min_area_ratio,
        )
    )
    checks.append(
        CheckRecord(
            name="edge_clearance", stage=2, raw_value=edge_clearance,
            threshold=edge_min, margin=edge_clearance - edge_min,
            passed=edge_clearance >= edge_min,
        )
    )

    passed = all(c.passed for c in checks)
    return StageResult(
        stage=2, name="grasp_suctionable", passed=passed, checks=checks,
        outputs={
            "plane_point": plane_point,
            "plane_normal": plane_normal,
            "rmse": rmse,
            "n_gripper_points": int(len(p_grip)),
            "gripper_width_m": grip.width_m,
            "gripper_length_m": grip.length_m,
        },
    )


def run_stage3(
    p_full: np.ndarray,
    p_g: np.ndarray,
    z_top: float,
    plane: tuple[float, float, float, float],
    cfg: dict,
    long_dir_xy: np.ndarray | None = None,
) -> StageResult:
    """Stage 3 - is the vertical lift corridor above the grasp clear?

    The corridor cross-section matches the (oriented) gripper footprint plus a
    safety margin: long side along ``long_dir_xy``, short side perpendicular.
    """
    s = cfg["stage3"]
    grip = resolve_gripper(cfg)
    safety = grip.safety_margin_m
    half_long = max(grip.half_w_m, grip.half_l_m)
    half_short = min(grip.half_w_m, grip.half_l_m)
    h_long = s.get("corridor_half_w_m")
    h_long = float(h_long) if h_long is not None else half_long + safety
    h_short = s.get("corridor_half_l_m")
    h_short = float(h_short) if h_short is not None else half_short + safety
    approach_h = resolve_corridor_height(cfg)
    top_band = float(s["top_band_m"])
    noise_tol = int(s["noise_point_tolerance"])
    corridor_reach = float(np.hypot(h_long, h_short))

    if z_top is None:
        z_top = float(heights_above_plane(np.asarray(p_g).reshape(1, 3), plane)[0])

    heights = heights_above_plane(p_full, plane)
    xy = project_to_plane_xy(p_full, plane)
    g_xy = project_to_plane_xy(np.asarray(p_g, dtype=np.float64).reshape(1, 3), plane)[0]
    rel = (xy - g_xy[None, :]) @ _gripper_rot(long_dir_xy)

    above = heights > (z_top + top_band)
    if not np.any(above):
        nearest_above_d = float("inf")
    else:
        nearest_above_d = float(np.linalg.norm(rel[above], axis=1).min())

    in_corridor = above & (np.abs(rel[:, 0]) <= h_long) & (np.abs(rel[:, 1]) <= h_short)
    n_block = int(in_corridor.sum())

    if n_block > 0:
        clearance_height = float(heights[in_corridor].min() - z_top)
    else:
        clearance_height = approach_h

    if np.isinf(nearest_above_d):
        reach_margin = _LARGE_MARGIN
    else:
        reach_margin = nearest_above_d - corridor_reach

    passed = n_block <= noise_tol
    checks = [
        CheckRecord(
            name="corridor_clear",
            stage=3,
            raw_value=float(
                nearest_above_d if not np.isinf(nearest_above_d) else corridor_reach + _LARGE_MARGIN
            ),
            threshold=corridor_reach,
            margin=reach_margin,
            passed=passed,
            detail={
                "n_blocking_points": n_block,
                "noise_tolerance": noise_tol,
                "corridor_half_long_m": h_long,
                "corridor_half_short_m": h_short,
                "gripper_width_m": grip.width_m,
                "gripper_length_m": grip.length_m,
                "clearance_height_m": clearance_height,
                "safety_corridor_height_m": approach_h,
                "z_top_m": float(z_top),
            },
        )
    ]
    return StageResult(
        stage=3, name="corridor_free", passed=passed, checks=checks,
        outputs={"n_blocking_points": n_block, "clearance_height_m": clearance_height},
    )
