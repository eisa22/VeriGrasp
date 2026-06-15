"""Tests for the Stage 13 grasp verification module.

Synthetic top-down scenes exercise each stage's reject path plus a clean
ACCEPT, and assert that the deterministic plane fit yields identical margins
across runs (reproducibility for the certifiability argument).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from perception.candidate import CandidateOut
from perception.extraction_corridor import compute_extraction_corridor
from Segmentation.pallet_scene import SessionContext
from verification import load_verification_config, verify_grasp
from verification.config import resolve_corridor_height
from verification.geometry import Intrinsics, target_pointcloud

FX = FY = 400.0
H, W = 240, 320
CX, CY = W / 2.0, H / 2.0
Z_PALLET = 1.2


def _session(depth: np.ndarray) -> SessionContext:
    return SessionContext(
        depth_abs=depth.astype(np.float32),
        depth_rel=depth.astype(np.float32),
        workspace_mask=np.ones((H, W), dtype=bool),
        plane_model=np.array([0.0, 0.0, 1.0, -Z_PALLET]),
        z_pallet_m=Z_PALLET,
        x_range=(0, W),
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )


def _candidate(
    bbox: tuple[int, int, int, int],
    cid: str = "v-01",
    top_h: float = Z_PALLET - 1.0,
) -> CandidateOut:
    x1, y1, x2, y2 = bbox
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return CandidateOut(
        candidate_id=cid,
        mask_2d=mask,
        points_3d=np.zeros((1, 3)),
        centroid_3d=np.zeros(3),
        surface_normal=np.array([0.0, 0.0, -1.0]),
        surface_area_m2=0.05,
        top_surface_height=top_h,
        bbox_2d=(x1, y1, x2, y2),
        debug={"label": "box"},
    )


def _grasp(row: int, col: int, z: float) -> SimpleNamespace:
    """Lightweight grasp stub (verify_grasp only reads position + rank)."""
    x = (col - CX) * z / FX
    y = (row - CY) * z / FY
    return SimpleNamespace(
        score=0.9,
        normal=np.array([0.0, 0.0, -1.0]),
        position=np.array([x, y, z]),
        row=row,
        col=col,
        rank=0,
    )


def _scene_with_box(z_top: float = 1.0, half: int = 40):
    """Pallet at Z_PALLET with a flat box top at z_top (smaller depth)."""
    depth = np.full((H, W), Z_PALLET, dtype=np.float32)
    cy, cx = H // 2, W // 2
    depth[cy - half : cy + half, cx - half : cx + half] = z_top
    bbox = (cx - half, cy - half, cx + half, cy + half)
    return depth, bbox, (cy, cx)


def _enrich_mask_points(cand: CandidateOut, sess: SessionContext) -> CandidateOut:
    intr = Intrinsics(fx=sess.fx, fy=sess.fy, cx=sess.cx, cy=sess.cy)
    pts = target_pointcloud(np.asarray(sess.depth_abs), cand.mask_2d, intr)
    if len(pts):
        cand.points_3d = pts
        cand.centroid_3d = pts.mean(axis=0)
    return cand


def _corridor_for(
    cand: CandidateOut,
    sess: SessionContext,
    *,
    half_long: float | None = None,
    half_short: float | None = None,
) -> dict:
    plane = tuple(float(x) for x in sess.plane_model)
    cfg = load_verification_config()
    corridor = compute_extraction_corridor(
        _enrich_mask_points(cand, sess),
        plane,
        lift_height_m=resolve_corridor_height(cfg),
    )
    assert corridor is not None
    if half_long is not None:
        corridor["corridor_half_long_m"] = half_long
    if half_short is not None:
        corridor["corridor_half_short_m"] = half_short
    return corridor


def test_clean_box_accept():
    depth, bbox, (r, c) = _scene_with_box()
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(r, c, 1.0)
    res = verify_grasp(grasp, cand, sess, corridor=_corridor_for(cand, sess))
    assert res.verdict == "ACCEPT", res.to_serializable()


def test_holey_bbox_rejects_stage1():
    depth, bbox, (r, c) = _scene_with_box()
    # Punch many holes inside the mask -> existence ratio collapses.
    x1, y1, x2, y2 = bbox
    depth[y1:y2:2, x1:x2] = 0.0
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(r, c, 1.0)
    res = verify_grasp(grasp, cand, sess, corridor=_corridor_for(cand, sess))
    assert res.verdict == "REJECT"
    assert res.decisive_stage == 1
    assert res.decisive_check == "existence"


def test_height_mismatch_rejects_stage1():
    depth, bbox, (r, c) = _scene_with_box()
    sess = _session(depth)
    # Pipeline claims a top height far from the real one (0.20 m) at the grasp.
    cand = _candidate(bbox, top_h=0.50)
    grasp = _grasp(r, c, 1.0)
    res = verify_grasp(
        grasp, cand, sess, corridor=_corridor_for(cand, sess)
    )
    assert res.verdict == "REJECT"
    assert res.decisive_stage == 1
    assert res.decisive_check == "top_height_match"


def test_empty_segmentation_rejects_stage1():
    depth = np.full((H, W), Z_PALLET, dtype=np.float32)
    cy, cx = H // 2, W // 2
    half = 40
    bbox = (cx - half, cy - half, cx + half, cy + half)
    # The segmented region carries no valid depth at all.
    depth[cy - half : cy + half, cx - half : cx + half] = 0.0
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(cy, cx, 1.0)
    res = verify_grasp(
        grasp, cand, sess, corridor=_corridor_for(cand, sess)
    )
    assert res.verdict == "REJECT"
    assert res.decisive_stage == 1
    assert res.decisive_check == "existence"


def test_grasp_in_hole_rejects_stage1():
    depth, bbox, (r, c) = _scene_with_box()
    # Hole exactly at the grasp point: window has too few points -> fail-closed.
    depth[r - 15 : r + 15, c - 15 : c + 15] = 0.0
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(r, c, 1.0)
    res = verify_grasp(
        grasp, cand, sess, corridor=_corridor_for(cand, sess)
    )
    assert res.verdict == "REJECT"
    assert res.decisive_stage == 1
    assert res.decisive_check == "top_height_match"


def test_tilted_surface_rejects_stage2():
    depth = np.full((H, W), Z_PALLET, dtype=np.float32)
    cy, cx = H // 2, W // 2
    half = 40
    # Steep tilt across columns -> surface normal far from vertical.
    cols = np.arange(W, dtype=np.float32)
    tilt = 0.004 * (cols - cx)  # ~58 deg at fx=400, z~1
    region = np.zeros((H, W), dtype=bool)
    region[cy - half : cy + half, cx - half : cx + half] = True
    base = 1.0
    depth_tilt = np.broadcast_to(base + tilt[None, :], (H, W))
    depth[region] = depth_tilt[region]
    bbox = (cx - half, cy - half, cx + half, cy + half)
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(cy, cx, float(depth[cy, cx]))
    # Isolate Stage 2: relax the Stage-1 height gate so the tilt (not the
    # height spread it induces) is the decisive reject.
    cfg = load_verification_config()
    cfg["stage1"]["top_height_tol_m"] = 1.0
    res = verify_grasp(
        grasp,
        cand,
        sess,
        config=cfg,
        corridor=_corridor_for(cand, sess),
    )
    assert res.verdict == "REJECT"
    assert res.decisive_stage == 2


def test_safety_corridor_height_default_is_30cm():
    from verification.config import resolve_corridor_height

    cfg = load_verification_config()
    assert resolve_corridor_height(cfg) >= 0.30


def test_neighbor_in_corridor_rejects_stage3():
    depth, bbox, (r, c) = _scene_with_box(half=18)
    # Tall neighbor beside the (small) box: outside bbox + cup, inside corridor.
    depth[r - 6 : r + 6, c + 22 : c + 40] = 0.9  # height 0.3 > z_top 0.2
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(r, c, 1.0)
    corridor = _corridor_for(cand, sess, half_long=0.08, half_short=0.08)
    res = verify_grasp(grasp, cand, sess, corridor=corridor)
    assert res.verdict == "REJECT", res.to_serializable()
    assert res.decisive_stage == 3


def test_determinism_full_mode():
    depth, bbox, (r, c) = _scene_with_box()
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(r, c, 1.0)
    cfg = load_verification_config()
    cfg["mode"] = "full"
    corridor = _corridor_for(cand, sess)
    res_a = verify_grasp(grasp, cand, sess, config=cfg, corridor=corridor)
    res_b = verify_grasp(grasp, cand, sess, config=cfg, corridor=corridor)
    margins_a = [c.margin for c in res_a.all_checks()]
    margins_b = [c.margin for c in res_b.all_checks()]
    assert margins_a == pytest.approx(margins_b, abs=0.0)
    assert res_a.soft_score == pytest.approx(res_b.soft_score, abs=0.0)
    # Full mode computes all three stages regardless of pass/fail.
    assert len(res_a.stages) == 3


def test_summary_serializable_omits_stage_outputs():
    depth, bbox, (r, c) = _scene_with_box()
    sess = _session(depth)
    cand = _candidate(bbox)
    corridor = _corridor_for(cand, sess)
    res = verify_grasp(
        _grasp(r, c, 1.0), cand, sess, corridor=corridor
    )
    summary = res.to_summary_serializable()
    assert "stages" not in summary
    assert summary["verdict"] == "ACCEPT"
    stage_json = res.stages[0].to_serializable()
    assert "outputs" not in stage_json


def test_stage3_corridor_endpoints_in_detail():
    depth, bbox, (r, c) = _scene_with_box()
    sess = _session(depth)
    cand = _candidate(bbox)
    corridor = _corridor_for(cand, sess)
    res = verify_grasp(
        _grasp(r, c, 1.0), cand, sess, corridor=corridor
    )
    st3 = next(s for s in res.stages if s.stage == 3)
    cc = next(c for c in st3.checks if c.name == "corridor_clear")
    detail = cc.detail
    assert len(detail["corners_bottom_3d"]) == 4
    assert len(detail["corners_top_3d"]) == 4
    assert detail["corridor_z_top_m"] > detail["z_bottom_m"]
