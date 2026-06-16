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
from verification.config import resolve_gripper
from verification.types import CheckRecord, StageResult

_LARGE_MARGIN = 1.0


def run_stage1(
    p_target: np.ndarray,
    n_valid: int,
    n_mask_px: int,
    p_g: np.ndarray,
    claimed_top_m: float,
    plane: tuple[float, float, float, float],
    cfg: dict,
) -> StageResult:
    """Stage 1 - does the segmentation match the raw point cloud?

    Two checks consume only the target mask points and the grasp point:
    1. ``existence`` - enough valid raw points actually lie under the mask
       (rejects a segmentation hallucinated over invalid / empty depth).
    2. ``top_height_match`` - the top height measured *locally at the grasp
       point* (robust percentile over a planar window) agrees with the height
       the pipeline reported (``candidate.top_surface_height``). This measured
       height is the lift reference exported as ``z_top`` for Stage 3.
    """
    s = cfg["stage1"]
    min_frac = float(s["min_valid_fraction"])
    min_pts = int(s["min_points"])
    win_r = float(s["window_radius_m"])
    pct = float(s["top_percentile"])
    tol = float(s["top_height_tol_m"])
    min_win_pts = int(s.get("min_window_points", 10))

    p_target = np.asarray(p_target, dtype=np.float64)
    checks: list[CheckRecord] = []

    # --- Check 1: existence (global over the mask) ---
    ratio = (n_valid / n_mask_px) if n_mask_px > 0 else 0.0
    checks.append(
        CheckRecord(
            name="existence",
            stage=1,
            raw_value=ratio,
            threshold=min_frac,
            margin=ratio - min_frac,
            passed=(ratio >= min_frac) and (n_valid >= min_pts),
            detail={"n_valid": int(n_valid), "n_mask_px": int(n_mask_px)},
        )
    )

    # --- Check 2: top_height_match (local window at the grasp point) ---
    z_top = None
    n_win = 0
    if p_target.shape[0] >= 1:
        xy = project_to_plane_xy(p_target, plane)
        g_xy = project_to_plane_xy(
            np.asarray(p_g, dtype=np.float64).reshape(1, 3), plane
        )[0]
        dist = np.linalg.norm(xy - g_xy[None, :], axis=1)
        win = dist <= win_r
        n_win = int(win.sum())
        if n_win >= min_win_pts:
            z_top = float(
                np.percentile(heights_above_plane(p_target[win], plane), pct)
            )

    diff = abs(z_top - claimed_top_m) if z_top is not None else float("inf")
    checks.append(
        CheckRecord(
            name="top_height_match",
            stage=1,
            raw_value=diff,
            threshold=tol,
            margin=(tol - diff) if np.isfinite(diff) else -_LARGE_MARGIN,
            passed=bool(np.isfinite(diff) and diff <= tol),
            detail={
                "z_top_meas_m": z_top,
                "claimed_top_m": float(claimed_top_m),
                "n_window_points": n_win,
                "window_radius_m": win_r,
            },
        )
    )

    passed = all(c.passed for c in checks)
    return StageResult(
        stage=1,
        name="segmentation_consistent",
        passed=passed,
        checks=checks,
        outputs={"z_top": z_top},
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


def _footprint_raster_layout(
    half_w: float,
    half_l: float,
    raster_m: float,
) -> tuple[int, int, np.ndarray, callable]:
    """Raster indices and footprint mask in gripper-local coordinates."""
    nx = max(1, int(np.ceil(2 * half_w / raster_m)))
    ny = max(1, int(np.ceil(2 * half_l / raster_m)))
    cx = (np.arange(nx) + 0.5) * raster_m - half_w
    cy = (np.arange(ny) + 0.5) * raster_m - half_l
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    required = (np.abs(gx) <= half_w) & (np.abs(gy) <= half_l)

    def to_cell(xy: np.ndarray) -> np.ndarray:
        ix = np.floor((xy[:, 0] + half_w) / raster_m).astype(int)
        iy = np.floor((xy[:, 1] + half_l) / raster_m).astype(int)
        return np.stack(
            [np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)],
            axis=1,
        )

    return nx, ny, required, to_cell


def _footprint_height_grid(
    p_grip: np.ndarray,
    rel_xy: np.ndarray,
    plane: tuple[float, float, float, float],
    half_w: float,
    half_l: float,
    raster_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Median height-above-plane per raster cell (NaN where empty)."""
    nx, ny, required, to_cell = _footprint_raster_layout(half_w, half_l, raster_m)
    grid = np.full((nx, ny), np.nan, dtype=np.float64)
    if len(p_grip) == 0:
        return grid, required

    heights = heights_above_plane(p_grip, plane)
    cells = to_cell(rel_xy)
    buckets: dict[tuple[int, int], list[float]] = {}
    for (i, j), h in zip(cells, heights):
        key = (int(i), int(j))
        buckets.setdefault(key, []).append(float(h))
    for (i, j), hs in buckets.items():
        grid[i, j] = float(np.median(hs))
    return grid, required


def _footprint_empty_fraction(
    rel_xy: np.ndarray,
    half_w: float,
    half_l: float,
    raster_m: float,
) -> float:
    """Fraction of footprint raster cells with no points (no morphological closing)."""
    _, _, required, to_cell = _footprint_raster_layout(half_w, half_l, raster_m)
    n_required = int(required.sum())
    if n_required == 0 or rel_xy.shape[0] == 0:
        return 1.0
    filled = np.zeros(required.shape, dtype=bool)
    cells = to_cell(rel_xy)
    filled[cells[:, 0], cells[:, 1]] = True
    n_filled = int((filled & required).sum())
    return 1.0 - (n_filled / n_required)


def _grid_seam_span(grid: np.ndarray, step_m: float) -> float:
    """Max row/column fraction of valid neighbours separated by a height step."""
    spans = [0.0]
    if grid.shape[1] >= 2:
        for i in range(grid.shape[0]):
            row = grid[i, :]
            diff = np.abs(row[1:] - row[:-1])
            with np.errstate(invalid="ignore"):
                exceed = diff > step_m
            valid = ~np.isnan(row[1:]) & ~np.isnan(row[:-1])
            denom = valid.sum()
            if denom > 0:
                spans.append(float(exceed[valid].sum()) / float(denom))
    if grid.shape[0] >= 2:
        for j in range(grid.shape[1]):
            col = grid[:, j]
            diff = np.abs(col[1:] - col[:-1])
            with np.errstate(invalid="ignore"):
                exceed = diff > step_m
            valid = ~np.isnan(col[1:]) & ~np.isnan(col[:-1])
            denom = valid.sum()
            if denom > 0:
                spans.append(float(exceed[valid].sum()) / float(denom))
    return max(spans)


def _footprint_depth_seam_span(
    p_grip: np.ndarray,
    rel_xy: np.ndarray,
    plane: tuple[float, float, float, float],
    half_w: float,
    half_l: float,
    raster_m: float,
    step_m: float,
) -> float:
    """Largest seam span across the gripper footprint height grid."""
    grid, _ = _footprint_height_grid(
        p_grip, rel_xy, plane, half_w, half_l, raster_m
    )
    return _grid_seam_span(grid, step_m)


def _footprint_peak_to_valley(
    p_grip: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    inlier_mask: np.ndarray | None = None,
) -> float:
    """Peak-to-valley signed residual range relative to the fitted plane."""
    pts = np.asarray(p_grip, dtype=np.float64)
    if pts.size == 0:
        return float("inf")
    if inlier_mask is not None and inlier_mask.any():
        pts = pts[inlier_mask]
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    centroid = np.asarray(plane_point, dtype=np.float64).reshape(3)
    residuals = (pts - centroid[None, :]) @ normal
    return float(residuals.max() - residuals.min())


def _footprint_peak_to_valley_robust(
    p_grip: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    low_pct: float = 2.5,
    high_pct: float = 97.5,
) -> float:
    """Outlier-robust peak-to-valley: percentile spread of signed residuals.

    Uses ``high_pct - low_pct`` of the signed plane residuals instead of the
    raw ``max - min`` so a single flying-pixel outlier no longer dominates the
    warp estimate (complements the non-robust ``_footprint_peak_to_valley``).
    """
    pts = np.asarray(p_grip, dtype=np.float64)
    if pts.size == 0:
        return float("inf")
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    centroid = np.asarray(plane_point, dtype=np.float64).reshape(3)
    residuals = (pts - centroid[None, :]) @ normal
    hi = float(np.percentile(residuals, high_pct))
    lo = float(np.percentile(residuals, low_pct))
    return hi - lo


def _holdability(
    p_g: np.ndarray,
    plane: tuple[float, float, float, float],
    parcel_obb: dict,
    pad_area_m2: float,
    cup_reff_m: float,
    density_kg_m3: float,
    vacuum_pressure_pa: float,
) -> tuple[float, float, float, float] | None:
    """Quasi-static suction wrench resistance (Dex-Net 3.0 style).

    Returns (hold_factor, safety_threshold_unused, drive_moment, resist_moment)
    where ``hold_factor = F_vac / W`` and the peel moments are in Nm. Returns
    None when the OBB lacks the extents/center needed to estimate mass and CoM.
    """
    try:
        extents = np.asarray(parcel_obb["extents"], dtype=np.float64).reshape(3)
        center = np.asarray(parcel_obb["center"], dtype=np.float64).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None

    g = 9.81
    volume = float(abs(extents[0] * extents[1] * extents[2]))
    mass = volume * float(density_kg_m3)
    weight = mass * g
    f_vac = float(vacuum_pressure_pa) * float(pad_area_m2)

    hold_factor = f_vac / weight if weight > 1e-9 else _LARGE_MARGIN

    g_xy = project_to_plane_xy(np.asarray(p_g, dtype=np.float64).reshape(1, 3), plane)[0]
    c_xy = project_to_plane_xy(center.reshape(1, 3), plane)[0]
    lever = float(np.linalg.norm(g_xy - c_xy))

    drive_moment = weight * lever
    resist_moment = f_vac * float(cup_reff_m)
    return hold_factor, weight, drive_moment, resist_moment


def run_stage2(
    p_target: np.ndarray,
    p_g: np.ndarray,
    plane: tuple[float, float, float, float],
    approach_axis: np.ndarray,
    cfg: dict,
    long_dir_xy: np.ndarray | None = None,
    grasp_normal: np.ndarray | None = None,
    parcel_obb: dict | None = None,
) -> StageResult:
    """Stage 2 - is the grasp point suctionable (planar, aligned, sealed)?

    The grasp position is the centre of the configured rectangular gripper
    footprint. ``p_target`` must contain only points on the selected parcel.
    ``long_dir_xy`` (pallet-plane unit vector) orients the gripper's long side
    along the parcel's longer side; None falls back to the u axis.
    ``grasp_normal`` (SuctionNet per-point normal) and ``parcel_obb`` (mass /
    CoM source) feed the additional physical checks; both are optional.
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
    max_empty_frac = float(s["max_empty_cell_fraction"])
    depth_seam_step = float(s["depth_seam_step_m"])
    depth_seam_span_max = float(s["depth_seam_span_ratio"])
    max_ptv = float(s["max_peak_to_valley_m"])
    # Additional physical checks (additive; bestehende Checks unveraendert).
    align_max = float(s["normal_alignment_max_deg"])
    max_ptv_robust = float(s["max_peak_to_valley_robust_m"])
    warp_low_pct = float(s["warp_robust_low_pct"])
    warp_high_pct = float(s["warp_robust_high_pct"])
    density = float(s["object_density_kg_m3"])
    vacuum_pa = float(s["vacuum_pressure_pa"])
    hold_safety = float(s["holdability_safety_factor"])
    pad_area = grip.width_m * grip.length_m
    cup_reff = 0.5 * min(grip.width_m, grip.length_m)
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
        checks.append(
            CheckRecord(
                name="data_gaps", stage=2, raw_value=1.0,
                threshold=max_empty_frac, margin=-_LARGE_MARGIN, passed=False,
            )
        )
        checks.append(
            CheckRecord(
                name="depth_seam", stage=2, raw_value=1.0,
                threshold=depth_seam_span_max, margin=-_LARGE_MARGIN, passed=False,
            )
        )
        checks.append(
            CheckRecord(
                name="surface_warp", stage=2, raw_value=float("inf"),
                threshold=max_ptv, margin=-_LARGE_MARGIN, passed=False,
            )
        )
        if grasp_normal is not None:
            checks.append(
                CheckRecord(
                    name="normal_alignment", stage=2, raw_value=90.0,
                    threshold=align_max, margin=-_LARGE_MARGIN, passed=False,
                )
            )
        checks.append(
            CheckRecord(
                name="surface_warp_robust", stage=2, raw_value=float("inf"),
                threshold=max_ptv_robust, margin=-_LARGE_MARGIN, passed=False,
            )
        )
        if parcel_obb is not None:
            checks.append(
                CheckRecord(
                    name="suction_force", stage=2, raw_value=0.0,
                    threshold=hold_safety, margin=-_LARGE_MARGIN, passed=False,
                )
            )
            checks.append(
                CheckRecord(
                    name="wrench_lever", stage=2, raw_value=float("inf"),
                    threshold=0.0, margin=-_LARGE_MARGIN, passed=False,
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

    empty_frac = _footprint_empty_fraction(rel_xy, half_long, half_short, raster_m)
    checks.append(
        CheckRecord(
            name="data_gaps", stage=2, raw_value=empty_frac,
            threshold=max_empty_frac, margin=max_empty_frac - empty_frac,
            passed=empty_frac <= max_empty_frac,
        )
    )

    seam_span = _footprint_depth_seam_span(
        p_grip, rel_xy, plane, half_long, half_short, raster_m, depth_seam_step
    )
    checks.append(
        CheckRecord(
            name="depth_seam", stage=2, raw_value=seam_span,
            threshold=depth_seam_span_max, margin=depth_seam_span_max - seam_span,
            passed=seam_span < depth_seam_span_max,
        )
    )

    peak_to_valley = _footprint_peak_to_valley(
        p_grip, plane_point, plane_normal
    )
    checks.append(
        CheckRecord(
            name="surface_warp", stage=2, raw_value=peak_to_valley,
            threshold=max_ptv, margin=max_ptv - peak_to_valley,
            passed=peak_to_valley <= max_ptv,
        )
    )

    # 2d normal_alignment (SuctionNet per-point normal vs fitted plane normal)
    if grasp_normal is not None:
        align = angle_between(grasp_normal, plane_normal)
        checks.append(
            CheckRecord(
                name="normal_alignment", stage=2, raw_value=align,
                threshold=align_max, margin=align_max - align,
                passed=align <= align_max,
            )
        )

    # 2e surface_warp_robust (percentile spread, outlier-robust)
    ptv_robust = _footprint_peak_to_valley_robust(
        p_grip, plane_point, plane_normal, warp_low_pct, warp_high_pct
    )
    checks.append(
        CheckRecord(
            name="surface_warp_robust", stage=2, raw_value=ptv_robust,
            threshold=max_ptv_robust, margin=max_ptv_robust - ptv_robust,
            passed=ptv_robust <= max_ptv_robust,
        )
    )

    # 2f holdability (suction wrench resistance) - only when OBB mass/CoM known
    if parcel_obb is not None:
        hold = _holdability(
            p_g, plane, parcel_obb, pad_area, cup_reff, density, vacuum_pa
        )
        if hold is not None:
            hold_factor, weight, drive_moment, resist_moment = hold
            checks.append(
                CheckRecord(
                    name="suction_force", stage=2, raw_value=hold_factor,
                    threshold=hold_safety, margin=hold_factor - hold_safety,
                    passed=hold_factor >= hold_safety,
                    detail={"weight_n": weight},
                )
            )
            moment_thr = resist_moment / hold_safety if hold_safety > 0 else resist_moment
            checks.append(
                CheckRecord(
                    name="wrench_lever", stage=2, raw_value=drive_moment,
                    threshold=moment_thr, margin=moment_thr - drive_moment,
                    passed=drive_moment <= moment_thr,
                    detail={"resist_moment_nm": resist_moment},
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
    corridor: dict,
    plane: tuple[float, float, float, float],
    cfg: dict,
) -> StageResult:
    """Stage 3 - collision check of the precomputed lift corridor vs. raw cloud.

    The corridor geometry is produced by the pipeline (``compute_extraction_corridor``);
    this stage only tests whether points in the full scene intersect the vertical
    lift volume above the package top.
    """
    s = cfg["stage3"]
    top_band = float(s["top_band_m"])
    noise_tol = int(s["noise_point_tolerance"])

    z_top = float(corridor.get("package_top_m", corridor.get("z_bottom_m", 0.0)))
    h_long = float(corridor["corridor_half_long_m"])
    h_short = float(corridor["corridor_half_short_m"])
    center = np.asarray(corridor["center_3d"], dtype=np.float64).reshape(3)
    long_dir = corridor.get("long_dir_xy")
    long_dir_xy = (
        np.asarray(long_dir, dtype=np.float64) if long_dir is not None else None
    )
    approach_h = float(
        corridor.get("safety_corridor_height_m", s.get("safety_corridor_height_m", 0.30))
    )
    corridor_reach = float(np.hypot(h_long, h_short))

    heights = heights_above_plane(p_full, plane)
    xy = project_to_plane_xy(p_full, plane)
    c_xy = project_to_plane_xy(center.reshape(1, 3), plane)[0]
    rel = (xy - c_xy[None, :]) @ _gripper_rot(long_dir_xy)

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
                "clearance_height_m": clearance_height,
                "safety_corridor_height_m": approach_h,
                "z_top_m": float(z_top),
                "z_bottom_m": float(corridor.get("z_bottom_m", z_top)),
                "corridor_z_top_m": float(
                    corridor.get("corridor_z_top_m", z_top + approach_h)
                ),
                "half_long_m": float(corridor.get("half_long_m", h_long)),
                "half_short_m": float(corridor.get("half_short_m", h_short)),
                "corners_bottom_3d": corridor.get("corners_bottom_3d"),
                "corners_top_3d": corridor.get("corners_top_3d"),
                "corridor_source": corridor.get("source"),
            },
        )
    ]
    return StageResult(
        stage=3, name="corridor_free", passed=passed, checks=checks,
        outputs={"n_blocking_points": n_block, "clearance_height_m": clearance_height},
    )
