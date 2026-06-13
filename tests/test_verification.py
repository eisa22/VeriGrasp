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
from Segmentation.pallet_scene import SessionContext
from verification import load_verification_config, verify_grasp

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


def _candidate(bbox: tuple[int, int, int, int], cid: str = "v-01") -> CandidateOut:
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
        top_surface_height=Z_PALLET - 1.0,
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


def test_clean_box_accept():
    depth, bbox, (r, c) = _scene_with_box()
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(r, c, 1.0)
    res = verify_grasp(grasp, cand, sess)
    assert res.verdict == "ACCEPT", res.to_serializable()


def test_holey_bbox_rejects_stage1():
    depth, bbox, (r, c) = _scene_with_box()
    # Punch many holes inside the bbox -> valid_ratio collapses.
    x1, y1, x2, y2 = bbox
    depth[y1:y2:2, x1:x2] = 0.0
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(r, c, 1.0)
    res = verify_grasp(grasp, cand, sess)
    assert res.verdict == "REJECT"
    assert res.decisive_stage == 1


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
    res = verify_grasp(grasp, cand, sess)
    assert res.verdict == "REJECT"
    assert res.decisive_stage == 2


def test_neighbor_in_corridor_rejects_stage3():
    depth, bbox, (r, c) = _scene_with_box(half=18)
    # Tall neighbor beside the (small) box: outside bbox + cup, inside corridor.
    depth[r - 6 : r + 6, c + 22 : c + 40] = 0.9  # height 0.3 > z_top 0.2
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(r, c, 1.0)
    cfg = load_verification_config()
    cfg["gripper"]["width_m"] = 0.036
    cfg["gripper"]["length_m"] = 0.036
    cfg["stage3"]["corridor_half_w_m"] = 0.08
    cfg["stage3"]["corridor_half_l_m"] = 0.08
    res = verify_grasp(grasp, cand, sess, config=cfg)
    assert res.verdict == "REJECT", res.to_serializable()
    assert res.decisive_stage == 3


def test_determinism_full_mode():
    depth, bbox, (r, c) = _scene_with_box()
    sess = _session(depth)
    cand = _candidate(bbox)
    grasp = _grasp(r, c, 1.0)
    cfg = load_verification_config()
    cfg["mode"] = "full"
    res_a = verify_grasp(grasp, cand, sess, config=cfg)
    res_b = verify_grasp(grasp, cand, sess, config=cfg)
    margins_a = [c.margin for c in res_a.all_checks()]
    margins_b = [c.margin for c in res_b.all_checks()]
    assert margins_a == pytest.approx(margins_b, abs=0.0)
    assert res_a.soft_score == pytest.approx(res_b.soft_score, abs=0.0)
    # Full mode computes all three stages regardless of pass/fail.
    assert len(res_a.stages) == 3
