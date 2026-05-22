"""Tests for Stage 11 suction grasp generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from perception.configs.load import load_suction_grasp_config
from perception.grasp_generation.normal_std_backend import (
    grid_sample,
    run_normal_std,
    _apply_mask_filter,
)
from perception.grasp_generation.camera import camera_from_shape
from perception.grasp_generation.stage import compute_suction_grasps
from perception.grasp_generation.types import SuctionGraspResult
from perception.selection.select_target import SelectedTarget, SelectionResult
from perception.candidate import CandidateOut
from Segmentation.pallet_scene import SessionContext


FX = FY = 437.04
H, W = 240, 320


def _flat_depth(z: float = 1.2) -> np.ndarray:
    depth = np.full((H, W), z, dtype=np.float32)
    depth[:20, :] = 0.0
    depth[-20:, :] = 0.0
    return depth


def _rect_mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _candidate(mask: np.ndarray, cid: str = "test-01") -> CandidateOut:
    ys, xs = np.nonzero(mask)
    z = 1.2
    pts = np.stack(
        [
            (xs - W / 2) * z / FX,
            (ys - H / 2) * z / FY,
            np.full(len(xs), z),
        ],
        axis=1,
    ).astype(np.float32)
    return CandidateOut(
        candidate_id=cid,
        mask_2d=mask,
        points_3d=pts,
        centroid_3d=pts.mean(axis=0),
        surface_normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        surface_area_m2=0.05,
        top_surface_height=0.1,
        bbox_2d=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        debug={"label": "box_a"},
    )


def _session(depth: np.ndarray) -> SessionContext:
    ws = np.ones((H, W), dtype=bool)
    return SessionContext(
        depth_abs=depth,
        depth_rel=depth.copy(),
        workspace_mask=ws,
        plane_model=np.array([0.0, 0.0, 1.0, -1.2]),
        z_pallet_m=1.2,
        x_range=(48, 272),
    )


def test_load_suction_grasp_config_merges_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "custom.yaml"
        path.write_text(yaml.dump({"max_grasps": 5, "min_score": 0.5}), encoding="utf-8")
        cfg = load_suction_grasp_config(path)
    assert cfg["max_grasps"] == 5
    assert cfg["min_score"] == 0.5
    assert cfg["backend"] == "normal_std"


def test_mask_filter_removes_outside_and_low_score():
    mask = _rect_mask(80, 160, 100, 220)
    scores = np.array([0.9, 0.1, 0.8], dtype=np.float32)
    rows = np.array([120, 10, 130], dtype=np.int32)
    cols = np.array([150, 150, 180], dtype=np.int32)
    s, r, c = _apply_mask_filter(scores, rows, cols, mask, min_score=0.3)
    assert len(s) == 2
    assert 10 not in r


def test_run_normal_std_finds_grasps_in_mask():
    depth = _flat_depth()
    mask = _rect_mask(60, 180, 80, 240)
    camera = camera_from_shape(H, W, fx=FX, fy=FY)
    cfg = load_suction_grasp_config()
    cfg["min_score"] = 0.01
    cfg["max_grasps"] = 10
    grasps, debug = run_normal_std(depth, mask, camera, cfg)
    assert len(grasps) >= 1
    for g in grasps:
        assert mask[g.row, g.col]
        assert g.score >= cfg["min_score"]
    assert debug["n_selected"] == len(grasps)


def test_min_separation_reduces_duplicates():
    depth = _flat_depth()
    mask = _rect_mask(60, 180, 80, 240)
    camera = camera_from_shape(H, W, fx=FX, fy=FY)
    cfg = load_suction_grasp_config()
    cfg["min_score"] = 0.01
    cfg["max_grasps"] = 20
    cfg["min_separation_m"] = 0.01
    dense, _ = run_normal_std(depth, mask, camera, cfg)
    cfg["min_separation_m"] = 0.5
    sparse, _ = run_normal_std(depth, mask, camera, cfg)
    assert len(dense) >= len(sparse)


def test_compute_suction_grasps_integration():
    depth = _flat_depth()
    mask = _rect_mask(70, 170, 90, 230)
    cand = _candidate(mask)
    sel = SelectionResult(
        primary=SelectedTarget(candidate=cand, rank=0, score=0.2),
        ranking=[],
    )
    result = compute_suction_grasps(sel, _session(depth))
    assert isinstance(result, SuctionGraspResult)
    assert result.candidate_id == cand.candidate_id
    assert result.backend == "normal_std"
    assert len(result.grasps) >= 1
    ser = result.to_serializable()
    assert ser["n_grasps"] == len(result.grasps)


def test_compute_suction_grasps_no_primary():
    result = compute_suction_grasps(
        SelectionResult(primary=None), _session(_flat_depth())
    )
    assert result.grasps == []
    assert result.debug.get("error") == "no_primary_selection"


def test_grid_sample_empty_map():
    tiny = np.zeros((5, 5), dtype=np.float32)
    s, r, c = grid_sample(tiny, down_rate=10, topk=10)
    assert len(s) == 0
