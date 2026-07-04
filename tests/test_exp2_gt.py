"""Experiment 2: GT derivations + GT self-test on a synthetic scene.

The self-test (acceptance criterion 9.1) feeds GT masks and GT-derived
quantities through the full metric path as if they were predictions; every
error must vanish and every match must be perfect.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.exp2_gt import derive_gt_geometry, load_gt_geometries
from evaluation.gt import VisibilityFilter
from experiments.exp2_grasp.evaluate import _load_eval_config, evaluate_scene

PLANE = (0.0, 0.0, 1.0, -2.5)  # pallet at z = 2.5, camera looks along +z


def _box_corners(center, dims_wlh, yaw_rad):
    """8 camera-frame corners of an upright box (canonical up = -z)."""
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


def _gt_object(obj_id, class_name, center, dims_wlh, yaw_rad, visible_pixels):
    return {
        "id": obj_id,
        "class_name": class_name,
        "center_camera_frame": list(center),
        "dimensions_wlh": list(dims_wlh),
        "yaw_world_rad": yaw_rad,
        "volume_m3": dims_wlh[0] * dims_wlh[1] * dims_wlh[2],
        "bbox_corners_camera_frame": _box_corners(center, dims_wlh, yaw_rad),
        "visible_pixels": visible_pixels,
    }


def _write_scene(tmp_path: Path, objects, mask_regions) -> Path:
    """Synthetic scene dir with GT json, instance mask, and prep context."""
    scene = tmp_path / "scene_900"
    scene.mkdir()
    h, w = 60, 80
    inst = np.full((h, w), -1, dtype=np.int32)
    for lid, (y0, y1, x0, x1) in mask_regions.items():
        inst[y0:y1, x0:x1] = lid
    np.save(scene / "instance_mask.npy", inst)
    with open(scene / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump({"objects": objects}, f)
    # All-true workspace mask as RLE (leading zero-run of length 0).
    with open(scene / "stage_prep_context.json", "w", encoding="utf-8") as f:
        json.dump({
            "plane_model": list(PLANE),
            "z_pallet_m": 2.5,
            "fx": 500.0, "fy": 500.0, "cx": 40.0, "cy": 30.0,
            "height": h, "width": w,
            "workspace_mask_rle": [0, h * w],
        }, f)
    return scene


def test_derive_gt_geometry_upright_box():
    dims = (0.3, 0.2, 0.15)  # long axis = w = 0.3 along x at yaw 0
    center = (0.1, 0.05, 2.5 - 0.075)  # bottom on the pallet
    obj = _gt_object(0, "Test Box", center, dims, yaw_rad=0.0, visible_pixels=100)
    geo = derive_gt_geometry(obj, PLANE)

    assert geo.h_top == pytest.approx(0.15)
    assert geo.h_bottom == pytest.approx(0.0, abs=1e-12)
    assert geo.height_m == pytest.approx(0.15)
    assert geo.footprint_long_m == pytest.approx(0.3, abs=1e-4)
    assert geo.footprint_short_m == pytest.approx(0.2, abs=1e-4)
    # Top-face normal must face the camera (n_z < 0) and be vertical.
    assert geo.top_normal[2] == pytest.approx(-1.0)
    assert np.linalg.norm(geo.top_normal) == pytest.approx(1.0)


def test_derive_gt_geometry_yawed_box_footprint():
    dims = (0.4, 0.1, 0.1)
    obj = _gt_object(0, "Yawed", (0.0, 0.0, 2.4), dims, yaw_rad=np.radians(35.0),
                     visible_pixels=100)
    geo = derive_gt_geometry(obj, PLANE)
    assert geo.footprint_long_m == pytest.approx(0.4, abs=1e-3)
    assert geo.footprint_short_m == pytest.approx(0.1, abs=1e-3)
    assert geo.footprint_aspect == pytest.approx(4.0, rel=1e-2)


def test_derive_gt_geometry_tilted_box_corners_by_height():
    """Corners of a rolled box must be split by height, not index order."""
    dims = (0.3, 0.2, 0.1)
    corners = np.asarray(_box_corners((0.0, 0.0, 2.4), dims, 0.0))
    # Roll 15 deg about the x-axis, then shuffle the corner order.
    a = np.radians(15.0)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    center = corners.mean(axis=0)
    rolled = (corners - center) @ R.T + center
    rng = np.random.default_rng(0)
    rolled = rolled[rng.permutation(8)]

    obj = {
        "id": 0, "class_name": "Tilted", "center_camera_frame": center.tolist(),
        "dimensions_wlh": list(dims), "yaw_world_rad": 0.0,
        "bbox_corners_camera_frame": rolled.tolist(), "visible_pixels": 100,
    }
    geo = derive_gt_geometry(obj, PLANE)
    assert geo.height_m == pytest.approx(0.1 * np.cos(a), abs=2e-3)
    # Normal tilted by 15 deg from vertical, still camera-facing.
    tilt = np.degrees(np.arccos(-geo.top_normal[2]))
    assert tilt == pytest.approx(15.0, abs=0.5)


def _rolled_obj(dims, roll_deg, center=(0.0, 0.0, 2.3)):
    corners = np.asarray(_box_corners(center, dims, 0.0))
    a = np.radians(roll_deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    c = corners.mean(axis=0)
    rolled = (corners - c) @ R.T + c
    rng = np.random.default_rng(1)
    rolled = rolled[rng.permutation(8)]
    return {
        "id": 0, "class_name": "Rolled", "center_camera_frame": c.tolist(),
        "dimensions_wlh": list(dims), "yaw_world_rad": 0.0,
        "bbox_corners_camera_frame": rolled.tolist(), "visible_pixels": 100,
    }


def test_top_normal_45deg_roll_no_face_from_highest_corners():
    """At 45 deg roll the 4 highest corners straddle an edge (no face); the

    normal must still come from a real box face, i.e. be exactly 45 deg from
    vertical instead of the previous SVD garbage near 90 deg."""
    geo = derive_gt_geometry(_rolled_obj((0.3, 0.2, 0.2), 45.0), PLANE)
    tilt = np.degrees(np.arccos(np.clip(-geo.top_normal[2], -1, 1)))
    assert tilt == pytest.approx(45.0, abs=0.5)
    assert geo.top_normal[2] < 0  # camera-facing


def test_top_normal_box_on_its_side():
    """Box rolled 90 deg: the up-facing face is a genuine face, normal vertical."""
    geo = derive_gt_geometry(_rolled_obj((0.3, 0.2, 0.1), 90.0), PLANE)
    assert geo.top_normal[2] == pytest.approx(-1.0, abs=1e-6)
    # Height extent now spans the former l-axis.
    assert geo.height_m == pytest.approx(0.2, abs=1e-6)


def test_top_normal_thin_box_steep_tilt():
    """Thin parcel at 70 deg: face selection must pick the most up-facing

    face (the thin edge face at 20 deg from vertical), never a ~90 deg one."""
    geo = derive_gt_geometry(_rolled_obj((0.3, 0.2, 0.01), 70.0), PLANE)
    tilt = np.degrees(np.arccos(np.clip(-geo.top_normal[2], -1, 1)))
    assert tilt == pytest.approx(20.0, abs=0.5)


def test_gt_self_test_synthetic_scene(tmp_path):
    """Full metric path with GT posing as predictions: zero errors everywhere."""
    objects = [
        _gt_object(0, "Test Box", (0.02, 0.01, 2.42), (0.3, 0.2, 0.16),
                   np.radians(25.0), visible_pixels=200),
        _gt_object(1, "Square Box", (-0.05, 0.03, 2.45), (0.15, 0.15, 0.1),
                   np.radians(46.0), visible_pixels=150),
    ]
    scene = _write_scene(
        tmp_path, objects,
        mask_regions={0: (10, 30, 10, 40), 1: (35, 55, 45, 70)},
    )
    eval_cfg = _load_eval_config()
    result = evaluate_scene(
        scene, eval_cfg, VisibilityFilter(mode="absolute", absolute_min=1),
        gt_self_test=True,
    )

    rows = result["candidate_rows"]
    assert len(rows) == 2
    for r in rows:
        assert abs(r["e_lat_mm"]) < 1e-6
        assert abs(r["e_top_mm_signed"]) < 1e-6
        assert abs(r["e_bottom_mm_signed"]) < 1e-6
        assert abs(r["ext_err_long_rel"]) < 1e-6
        assert abs(r["ext_err_short_rel"]) < 1e-6
        assert abs(r["ext_err_height_rel"]) < 1e-6
        assert abs(r["yaw_err_deg"]) < 1e-6
        assert r["match_iou"] == pytest.approx(1.0)

    grasp = result["grasp_row"]
    assert grasp["status"] == "evaluated"
    assert abs(float(grasp["theta_deg"])) < 1e-6
    assert grasp["within_12deg"] is True and grasp["within_30deg"] is True

    # Near-square footprint must use the 90-deg fold.
    square_row = next(r for r in rows if r["gt_class"] == "Square Box")
    assert square_row["yaw_fold_deg"] == 90


def test_load_gt_geometries_skips_negative_ids(tmp_path):
    objects = [_gt_object(0, "Box", (0.0, 0.0, 2.4), (0.2, 0.1, 0.1), 0.0, 50)]
    scene = _write_scene(tmp_path, objects, mask_regions={0: (5, 20, 5, 25)})
    geos = load_gt_geometries(scene, PLANE)
    assert set(geos.keys()) == {0}
