"""Unit tests for Stage 2.5 bottom-plane inference.

All heights are in 'above pallet' convention (z_0 = 0 = pallet,
positive = above pallet, closer to camera).

PALLET_PLANE = (0, 0, 1, 0) means the pallet plane is z=0 in camera
coordinates. Camera-Z = -height_above_pallet, so test points are placed
at *negative* z-values to be 'above' the pallet.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.bottom_inference import infer_bottom_planes
from perception.candidate import CandidateOut
from perception.configs.load import load_bottom_inference_config


PALLET_PLANE = (0.0, 0.0, 1.0, 0.0)
CONFIG = load_bottom_inference_config()


def _make_mask(size: int = 32) -> np.ndarray:
    m = np.zeros((64, 64), dtype=np.uint8)
    m[16:16 + size, 16:16 + size] = 1
    return m


def _make_flat_parcel(
    candidate_id: str,
    top_h: float,
    center_xy: tuple[float, float],
    footprint: float = 0.10,
    grid: int = 20,
    noise_low: float | None = None,
    noise_count: int = 0,
) -> CandidateOut:
    """Flat top parcel at height top_h with optional low-noise outliers."""
    cx, cy = center_xy
    xs = np.linspace(cx - footprint / 2, cx + footprint / 2, grid)
    ys = np.linspace(cy - footprint / 2, cy + footprint / 2, grid)
    xx, yy = np.meshgrid(xs, ys)
    heights = np.full_like(xx, top_h, dtype=np.float64)
    pts_top = np.stack([xx.ravel(), yy.ravel(), -heights.ravel()], axis=1)

    if noise_low is not None and noise_count > 0:
        rng = np.random.default_rng(seed=hash(candidate_id) % 2**32)
        nx = rng.uniform(cx - footprint / 2, cx + footprint / 2, noise_count)
        ny = rng.uniform(cy - footprint / 2, cy + footprint / 2, noise_count)
        nz = -np.full(noise_count, noise_low)
        noise = np.stack([nx, ny, nz], axis=1)
        points = np.vstack([pts_top, noise])
    else:
        points = pts_top

    centroid = points.mean(axis=0)
    return CandidateOut(
        candidate_id=candidate_id,
        mask_2d=_make_mask(),
        points_3d=points,
        centroid_3d=centroid,
        surface_normal=np.array([0.0, 0.0, 1.0]),
        surface_area_m2=0.01,
        top_surface_height=top_h,
        bbox_2d=(10, 10, 50, 50),
    )


def _scene_floor(height: float, area_xy: tuple[float, float, float, float], n: int = 600) -> np.ndarray:
    """Dense scene point patch at the given height covering area_xy."""
    x0, y0, x1, y1 = area_xy
    rng = np.random.default_rng(seed=42)
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(x0, x1, side)
    ys = np.linspace(y0, y1, side)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel(), np.full(side * side, -height)], axis=1)


def test_solid_surface_ignores_noise():
    """A few low-noise points below the top must NOT pull the box down."""
    c = _make_flat_parcel("a", top_h=0.30, center_xy=(0.0, 0.0),
                          noise_low=0.05, noise_count=5)
    scene = np.zeros((10, 3))
    out = infer_bottom_planes([c], scene, PALLET_PLANE, CONFIG)
    assert out[0].debug["z_visible_min"] == pytest.approx(0.30, abs=0.01)


def test_solid_surface_detects_real_lower_plane():
    """A dense lower surface in the candidate's own points IS detected."""
    top = _make_flat_parcel("a", top_h=0.50, center_xy=(0.0, 0.0))
    lower = _make_flat_parcel("a_lower", top_h=0.30, center_xy=(0.0, 0.0))
    pts = np.vstack([top.points_3d, lower.points_3d])
    c = CandidateOut(
        candidate_id="combo",
        mask_2d=_make_mask(),
        points_3d=pts,
        centroid_3d=pts.mean(axis=0),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        surface_area_m2=0.02,
        top_surface_height=0.50,
        bbox_2d=(10, 10, 50, 50),
    )
    scene = np.zeros((10, 3))
    out = infer_bottom_planes([c], scene, PALLET_PLANE, CONFIG)
    assert out[0].debug["z_visible_min"] == pytest.approx(0.30, abs=0.01)


def test_scene_pcd_pulls_box_to_neighbor_below():
    """Top parcel detected, lower parcel only visible in scene_pcd -> box extends down."""
    top = _make_flat_parcel("top", top_h=0.62, center_xy=(0.0, 0.0), footprint=0.10)
    scene = _scene_floor(height=0.30, area_xy=(-0.08, -0.08, 0.08, 0.08), n=600)
    out = infer_bottom_planes([top], scene, PALLET_PLANE, CONFIG)
    assert out[0].bottom.bottom_method == "from_neighbor"
    assert out[0].bottom.bottom_z == pytest.approx(0.30, abs=0.01)
    assert out[0].bottom.height_m == pytest.approx(0.32, abs=0.02)
    assert "scene" in out[0].debug["case_label"]


def test_lateral_candidate_neighbor():
    """Lower DETECTED candidate also acts as neighbour."""
    top = _make_flat_parcel("top", top_h=0.62, center_xy=(0.0, 0.0))
    n1 = _make_flat_parcel("n1", top_h=0.30, center_xy=(0.12, 0.0))
    scene = np.zeros((10, 3))
    out = infer_bottom_planes([top, n1], scene, PALLET_PLANE, CONFIG)
    top_out = next(c for c in out if c.candidate_id == "top")
    assert top_out.bottom.bottom_method == "from_neighbor"
    assert top_out.bottom.bottom_z == pytest.approx(0.30, abs=0.01)


def test_fallback_close_to_pallet():
    """Isolated parcel close to pallet drops to pallet."""
    near = _make_flat_parcel("solo_near", top_h=0.32, center_xy=(0.0, 0.0))
    near_pts = near.points_3d.copy()
    near_pts[:5, 2] = -0.02
    near = CandidateOut(
        candidate_id=near.candidate_id,
        mask_2d=near.mask_2d,
        points_3d=near_pts,
        centroid_3d=near_pts.mean(axis=0),
        surface_normal=near.surface_normal,
        surface_area_m2=near.surface_area_m2,
        top_surface_height=0.32,
        bbox_2d=near.bbox_2d,
    )
    out = infer_bottom_planes([near], np.zeros((10, 3)), PALLET_PLANE, CONFIG)
    method = out[0].bottom.bottom_method
    assert method in ("from_pallet", "from_neighbor", "measured")
    assert out[0].bottom.bottom_z <= 0.35


def test_fallback_isolated_high():
    """Isolated parcel high above pallet, no scene below -> uncertain."""
    far = _make_flat_parcel("solo_far", top_h=0.80, center_xy=(0.5, 0.5))
    out = infer_bottom_planes([far], np.zeros((10, 3)), PALLET_PLANE, CONFIG)
    assert out[0].bottom.bottom_method == "uncertain"
    assert out[0].debug["case_label"] == "Fallback_none"


def test_obb_extrusion():
    top_h = 0.50
    bottom_h = 0.30
    c = _make_flat_parcel("obb", top_h=top_h, center_xy=(0.0, 0.0), footprint=0.12)
    cfg = dict(CONFIG)
    cfg["obb"] = {"min_points": 10, "max_aspect_ratio": 20.0}

    from perception.bottom_inference.obb import fit_extruded_obb, obb_xy_footprint_area

    obb = fit_extruded_obb(c, PALLET_PLANE, bottom_h, top_h, cfg)
    ext = np.asarray(obb["extents"])
    assert ext[2] == pytest.approx(top_h - bottom_h, abs=0.001)

    xy = c.points_3d[:, :2]
    mask_area = (xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min())
    obb_area = obb_xy_footprint_area(obb)
    rel_diff = abs(obb_area - mask_area) / max(mask_area, 1e-6)
    assert rel_diff <= 0.05


def test_immutability_and_audit_fields():
    c0 = _make_flat_parcel("a", top_h=0.30, center_xy=(-0.05, 0.0))
    c1 = _make_flat_parcel("b", top_h=0.30, center_xy=(0.05, 0.0))
    mask_ref = c0.mask_2d
    pts_ref = c0.points_3d
    out = infer_bottom_planes([c0, c1], np.zeros((10, 3)), PALLET_PLANE, CONFIG)
    assert out[0].mask_2d is mask_ref
    assert out[0].points_3d is pts_ref
    for c in out:
        assert c.bottom is not None
        for key in (
            "n_neighbors_found",
            "n_scene_points_below",
            "neighbor_ids_used",
            "z_visible_min",
            "z_highest_neighbor",
            "neighbor_source",
            "case_label",
        ):
            assert key in c.debug
