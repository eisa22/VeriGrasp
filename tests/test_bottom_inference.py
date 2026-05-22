"""Unit tests for Stage 2.5 bottom-plane inference (gradient-only).

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

# Synthetic image dimensions and pin-hole intrinsics matching FX=FY=437.04.
H, W = 240, 320
CX, CY = W / 2.0, H / 2.0


def _make_mask(size: int = 32, origin: tuple[int, int] = (16, 16)) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    oy, ox = origin
    m[oy:oy + size, ox:ox + size] = 1
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
    """Flat top parcel at height top_h with optional low-noise outliers.

    NOTE: mask_2d, points_3d and the synthetic 'image' (used by the gradient
    test) are independent here. We only rely on points_3d for solid_surface
    detection (lowest visible plane) which doesn't need a real image.
    """
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


def _build_synthetic_scene(target_top_h: float, neighbor_top_h: float):
    """Build (depth, sobel_edges, target_candidate, workspace_mask).

    Layout:
      - Image WxH = 320x240, pinhole FX=FY=437.04, cx=W/2, cy=H/2.
      - Target square mask at the centre (height target_top_h above pallet).
      - A neighbour square mask offset to the right at neighbor_top_h.
      - Sobel edges along the borders of both masks (where depths jump).
      - Pallet (depth ~5m) fills the rest of the workspace.

    Camera-Z (depth_abs) = z_pallet - height_above_pallet, with z_pallet = 5.0m
    chosen large to keep the pinhole projection reasonable.
    """
    z_pallet = 5.0
    depth = np.full((H, W), z_pallet, dtype=np.float64)
    edges = np.zeros((H, W), dtype=np.uint8)

    target_mask = np.zeros((H, W), dtype=np.uint8)
    tx0, ty0, tw = 130, 90, 60
    target_mask[ty0:ty0 + tw, tx0:tx0 + tw] = 1
    depth[ty0:ty0 + tw, tx0:tx0 + tw] = z_pallet - target_top_h
    edges[ty0 - 1:ty0 + tw + 1, tx0 - 1] = 1
    edges[ty0 - 1:ty0 + tw + 1, tx0 + tw] = 1
    edges[ty0 - 1, tx0 - 1:tx0 + tw + 1] = 1
    edges[ty0 + tw, tx0 - 1:tx0 + tw + 1] = 1

    nx0, ny0, nw = 210, 90, 60
    depth[ny0:ny0 + nw, nx0:nx0 + nw] = z_pallet - neighbor_top_h
    edges[ny0 - 1:ny0 + nw + 1, nx0 - 1] = 1
    edges[ny0 - 1:ny0 + nw + 1, nx0 + nw] = 1
    edges[ny0 - 1, nx0 - 1:nx0 + nw + 1] = 1
    edges[ny0 + nw, nx0 - 1:nx0 + nw + 1] = 1

    workspace = np.ones((H, W), dtype=bool)

    fx = fy = 437.04
    ys, xs = np.where(target_mask > 0)
    z = depth[ys, xs]
    x = (xs - W / 2.0) * z / fx
    y = (ys - H / 2.0) * z / fy
    pts = np.stack([x, y, z], axis=1)

    target = CandidateOut(
        candidate_id="target",
        mask_2d=target_mask,
        points_3d=pts,
        centroid_3d=pts.mean(axis=0),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        surface_area_m2=0.01,
        top_surface_height=target_top_h,
        bbox_2d=(tx0, ty0, tx0 + tw, ty0 + tw),
    )

    # NOTE the test plane is offset so 'above pallet' matches z_pallet=5.0m:
    plane = (0.0, 0.0, 1.0, -z_pallet)
    return depth, edges, workspace, target, plane


def _relax_gradient_cfg(cfg: dict) -> dict:
    """Synthetic scene is 320x240; production defaults assume HD. Loosen filters."""
    cfg["gradient_neighbor"]["neighbor_radius_m"] = None
    cfg["gradient_neighbor"]["neighbor_radius_px"] = 100
    cfg["gradient_neighbor"]["min_plateau_area_px"] = 200
    cfg["gradient_neighbor"]["min_plateau_area_m2"] = 0.0
    cfg["gradient_neighbor"]["min_aspect_ratio"] = 0.1
    cfg["gradient_neighbor"]["max_plateau_z_std_m"] = 0.05
    return cfg


def test_gradient_neighbor_pulls_box_down():
    """Lower plateau in the neighbourhood is detected via Sobel edges."""
    depth, edges, ws, target, plane = _build_synthetic_scene(
        target_top_h=0.62, neighbor_top_h=0.30,
    )
    cfg = _relax_gradient_cfg(load_bottom_inference_config())
    out = infer_bottom_planes(
        [target], plane, cfg,
        depth=depth, sobel_edges=edges, workspace_mask=ws,
    )
    assert out[0].bottom.bottom_method == "from_neighbor"
    assert out[0].bottom.bottom_z == pytest.approx(0.30, abs=0.02)
    assert out[0].debug["neighbor_source"] == "gradient"
    assert out[0].debug["gradient_n_plateaus_kept"] >= 1


def test_gradient_neighbor_picks_pallet_when_same_height_neighbor():
    """
    Target and neighbour have the same height; pallet visible around them.
    The pallet (a deeper plateau) is picked as the neighbour -> box drops
    to the pallet level.
    """
    depth, edges, ws, target, plane = _build_synthetic_scene(
        target_top_h=0.30, neighbor_top_h=0.30,
    )
    cfg = _relax_gradient_cfg(load_bottom_inference_config())
    out = infer_bottom_planes(
        [target], plane, cfg,
        depth=depth, sobel_edges=edges, workspace_mask=ws,
    )
    # Same-height neighbour is filtered out (>= z_visible_min - tol);
    # the surrounding pallet at h=0 wins.
    assert out[0].bottom.bottom_z == pytest.approx(0.0, abs=0.02)
    assert out[0].debug["neighbor_source"] == "gradient"


def test_no_gradient_data_falls_back_to_lateral():
    """Without depth/sobel data, lateral candidate neighbour is still used."""
    top = _make_flat_parcel("top", top_h=0.62, center_xy=(0.0, 0.0))
    n1 = _make_flat_parcel("n1", top_h=0.30, center_xy=(0.12, 0.0))
    out = infer_bottom_planes([top, n1], PALLET_PLANE, CONFIG)
    top_out = next(c for c in out if c.candidate_id == "top")
    assert top_out.bottom.bottom_method == "from_neighbor"
    assert top_out.bottom.bottom_z == pytest.approx(0.30, abs=0.01)


def test_higher_of_gradient_and_lateral_wins():
    """When both sources produce a valid neighbour, the higher one wins."""
    depth, edges, ws, target, plane = _build_synthetic_scene(
        target_top_h=0.62, neighbor_top_h=0.10,
    )
    n_high = _make_flat_parcel("n_high", top_h=0.45, center_xy=(0.05, 0.0))
    cfg = _relax_gradient_cfg(load_bottom_inference_config())

    out = infer_bottom_planes(
        [target, n_high], plane, cfg,
        depth=depth, sobel_edges=edges, workspace_mask=ws,
    )
    t_out = next(c for c in out if c.candidate_id == "target")
    assert t_out.bottom.bottom_z == pytest.approx(0.45, abs=0.02)
    assert t_out.debug["neighbor_source"] == "lateral"


def test_lateral_neighbor_above_z_visible_min_is_ignored():
    """A neighbour whose top is ABOVE z_visible_min must not be used."""
    target = _make_flat_parcel("target", top_h=0.30, center_xy=(0.0, 0.0))
    blocker = _make_flat_parcel("blocker", top_h=0.50, center_xy=(0.12, 0.0))
    out = infer_bottom_planes([target, blocker], PALLET_PLANE, CONFIG)
    t_out = next(c for c in out if c.candidate_id == "target")
    assert t_out.debug["z_neighbor_top_lateral"] is None


def test_solid_surface_ignores_noise():
    """A few low-noise points must NOT pull z_visible_min down."""
    c = _make_flat_parcel("a", top_h=0.30, center_xy=(0.0, 0.0),
                          noise_low=0.05, noise_count=5)
    out = infer_bottom_planes([c], PALLET_PLANE, CONFIG)
    assert out[0].debug["z_visible_min"] == pytest.approx(0.30, abs=0.01)


def test_fallback_isolated_high():
    """Isolated parcel high above pallet, no neighbours -> uncertain."""
    far = _make_flat_parcel("solo_far", top_h=0.80, center_xy=(0.5, 0.5))
    out = infer_bottom_planes([far], PALLET_PLANE, CONFIG)
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
    out = infer_bottom_planes([c0, c1], PALLET_PLANE, CONFIG)
    assert out[0].mask_2d is mask_ref
    assert out[0].points_3d is pts_ref
    for c in out:
        assert c.bottom is not None
        for key in (
            "n_lateral_neighbors",
            "neighbor_ids_used",
            "z_visible_min",
            "z_neighbor_top_gradient",
            "z_neighbor_top_lateral",
            "z_highest_neighbor",
            "neighbor_source",
            "case_label",
            "gradient_n_plateaus_kept",
            "gradient_n_components_total",
            "gradient_rejections",
        ):
            assert key in c.debug
