"""Experiment 3: grasp-validity oracle unit tests and import independence."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.exp2_gt import derive_gt_geometry
from evaluation.exp3_oracle import (
    OracleParams,
    evaluate_oracle,
    load_gt_corner_boxes,
)
from perception.geometry.plane import (
    heights_above_plane,
    project_to_plane_xy,
    unproject_from_plane_xy,
)

PLANE = (0.0, 0.0, 1.0, -2.5)


def _box_corners(center, dims_wlh, yaw_rad=0.0):
    w, l, h = dims_wlh
    cx, cy, cz = center
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    corners = []
    for dx in (-w / 2, w / 2):
        for dy in (-l / 2, l / 2):
            x = cx + c * dx - s * dy
            y = cy + s * dx + c * dy
            for dz in (-h / 2, h / 2):
                corners.append([x, y, cz + dz])
    return corners


def _gt_object(obj_id, class_name, center, dims_wlh, yaw_rad=0.0):
    return {
        "id": obj_id,
        "class_name": class_name,
        "center_camera_frame": list(center),
        "dimensions_wlh": list(dims_wlh),
        "yaw_world_rad": yaw_rad,
        "volume_m3": dims_wlh[0] * dims_wlh[1] * dims_wlh[2],
        "bbox_corners_camera_frame": _box_corners(center, dims_wlh, yaw_rad),
        "visible_pixels": 500,
    }


def _top_grasp(obj, xy_offset=(0.0, 0.0), z_offset_m: float = 0.0):
    """Grasp point on the GT top face (optionally shifted in-plane / height)."""
    corners = np.asarray(obj["bbox_corners_camera_frame"], dtype=np.float64)
    geo = derive_gt_geometry(obj, PLANE)
    heights = heights_above_plane(corners, PLANE)
    top = corners[np.argsort(-heights)[:4]]
    top_xy = project_to_plane_xy(top, PLANE).mean(axis=0)
    top_xy = top_xy + np.asarray(xy_offset, dtype=np.float64)
    grasp = unproject_from_plane_xy(
        top_xy.reshape(1, 2), PLANE, heights=geo.h_top + z_offset_m
    )[0]
    return grasp, geo


def _run_oracle(obj, grasp_pos, grasp_normal, *, extra_objects=None, yaw_deg=0.0):
    objects = [obj] + (extra_objects or [])
    gt_geos = {}
    all_corners = {}
    for o in objects:
        geo = derive_gt_geometry(o, PLANE)
        gt_geos[geo.instance_id] = geo
        all_corners[geo.instance_id] = np.asarray(
            o["bbox_corners_camera_frame"], dtype=np.float64
        )
    matched = gt_geos[0]
    return evaluate_oracle(
        grasp_position=np.asarray(grasp_pos, dtype=np.float64),
        grasp_normal=np.asarray(grasp_normal, dtype=np.float64),
        pred_yaw_deg=yaw_deg,
        target_matched=True,
        matched_gt=matched,
        matched_corners=all_corners[0],
        all_gt_geos=gt_geos,
        all_corners=all_corners,
        plane=PLANE,
        params=OracleParams(),
    )


def test_oracle_valid_centered_aligned():
    dims = (0.30, 0.20, 0.15)
    center = (0.0, 0.0, 2.5 - 0.075)
    obj = _gt_object(0, "Box", center, dims)
    grasp, geo = _top_grasp(obj)
    label = _run_oracle(obj, grasp, [0.0, 0.0, -1.0], yaw_deg=geo.footprint_yaw_deg)
    assert label.valid is True
    assert label.violated == []


def test_oracle_surface_contact_violated():
    dims = (0.30, 0.20, 0.15)
    center = (0.0, 0.0, 2.5 - 0.075)
    obj = _gt_object(0, "Box", center, dims)
    # 40 mm above the top face along the approach axis (toward camera).
    grasp, _ = _top_grasp(obj, z_offset_m=0.040)
    label = _run_oracle(obj, grasp, [0.0, 0.0, -1.0], yaw_deg=0.0)
    assert label.valid is False
    assert "surface_contact" in label.violated


def test_oracle_footprint_containment_violated():
    dims = (0.30, 0.20, 0.15)
    center = (0.0, 0.0, 2.5 - 0.075)
    obj = _gt_object(0, "Box", center, dims)
    grasp, _ = _top_grasp(obj, xy_offset=(0.20, 0.0))
    label = _run_oracle(obj, grasp, [0.0, 0.0, -1.0], yaw_deg=0.0)
    assert label.valid is False
    assert "footprint_containment" in label.violated


def test_oracle_free_extraction_violated():
    dims = (0.30, 0.20, 0.15)
    center = (0.0, 0.0, 2.5 - 0.075)
    obj = _gt_object(0, "Box", center, dims)
    grasp, _ = _top_grasp(obj)
    # Blocker overlapping the lift corridor above the target (same XY footprint).
    blocker = _gt_object(
        1,
        "Blocker",
        (0.0, 0.0, 2.25),
        (0.28, 0.18, 0.08),
    )
    label = _run_oracle(
        obj, grasp, [0.0, 0.0, -1.0],
        extra_objects=[blocker],
        yaw_deg=0.0,
    )
    assert label.valid is False
    assert "free_extraction" in label.violated


def test_oracle_normal_alignment_violated():
    dims = (0.30, 0.20, 0.15)
    center = (0.0, 0.0, 2.5 - 0.075)
    obj = _gt_object(0, "Box", center, dims)
    grasp, _ = _top_grasp(obj)
    tilt = np.radians(20.0)
    normal = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    label = _run_oracle(obj, grasp, normal, yaw_deg=0.0)
    assert label.valid is False
    assert "normal_alignment" in label.violated


def test_oracle_single_object_seam_violated():
    dims = (0.30, 0.20, 0.15)
    left = _gt_object(0, "Left", (-0.16, 0.0, 2.5 - 0.075), dims)
    right = _gt_object(1, "Right", (0.16, 0.0, 2.5 - 0.075), dims)
    # Grasp on the seam between the two boxes (adjacent along plane v).
    grasp, _ = _top_grasp(left, xy_offset=(0.0, -0.16))
    gt_geos = {
        0: derive_gt_geometry(left, PLANE),
        1: derive_gt_geometry(right, PLANE),
    }
    corners = {
        0: np.asarray(left["bbox_corners_camera_frame"], dtype=np.float64),
        1: np.asarray(right["bbox_corners_camera_frame"], dtype=np.float64),
    }
    label = evaluate_oracle(
        grasp_position=grasp,
        grasp_normal=np.array([0.0, 0.0, -1.0]),
        pred_yaw_deg=0.0,
        target_matched=True,
        matched_gt=gt_geos[0],
        matched_corners=corners[0],
        all_gt_geos=gt_geos,
        all_corners=corners,
        plane=PLANE,
        params=OracleParams(),
    )
    assert label.valid is False
    assert "single_object" in label.violated


def test_oracle_target_unmatched():
    label = evaluate_oracle(
        grasp_position=np.zeros(3),
        grasp_normal=np.array([0.0, 0.0, -1.0]),
        pred_yaw_deg=0.0,
        target_matched=False,
        matched_gt=None,
        matched_corners=None,
        all_gt_geos={},
        all_corners={},
        plane=PLANE,
        params=OracleParams(),
    )
    assert label.valid is False
    assert label.violated == ["single_object"]


def test_oracle_imports_are_independent():
    path = Path(__file__).resolve().parents[1] / "evaluation" / "exp3_oracle.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned = (
        "verification.stages",
        "verification.verify",
        "verification.box_check",
        "load_session_depth",
        "depth_abs",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for b in banned:
                    assert b not in alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for b in banned:
                assert b not in mod


def test_load_gt_corner_boxes(tmp_path):
    scene = tmp_path / "scene_001"
    scene.mkdir()
    obj = _gt_object(0, "Box", (0, 0, 2.4), (0.2, 0.2, 0.1))
    with open(scene / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump({"objects": [obj]}, f)
    corners = load_gt_corner_boxes(scene)
    assert 0 in corners
    assert corners[0].shape == (8, 3)
