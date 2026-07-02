"""Unit tests for Experiment 1 GT loading and visibility filtering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.gt import VisibilityFilter, build_gt_scene


def _write_scene(tmp_path: Path, objects: list[dict], mask_labels: dict[int, list[tuple[int, int]]]) -> Path:
    scene = tmp_path / "scene_test"
    scene.mkdir()
    h, w = 40, 40
    inst = np.full((h, w), -1, dtype=np.int32)
    for lid, pixels in mask_labels.items():
        for y, x in pixels:
            inst[y, x] = lid
    np.save(scene / "instance_mask.npy", inst)
    with open(scene / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump({"objects": objects}, f)
    return scene


def test_zero_visible_json_only_counts_as_invisible(tmp_path):
    scene = _write_scene(
        tmp_path,
        [
            {"id": 0, "class_name": "box", "visible_pixels": 4},
            {"id": 1, "class_name": "hidden", "visible_pixels": 0},
        ],
        {0: [(5, 5), (5, 6), (6, 5), (6, 6)]},
    )
    ws = np.ones((40, 40), dtype=bool)
    gt = build_gt_scene(scene, ws, visibility=VisibilityFilter(mode="absolute", absolute_min=1))
    assert gt.gt_total == 2
    assert len(gt.gt_eval) == 1
    assert gt.gt_invisible == 1
    assert gt.gt_out_of_scope == 0


def test_reconciliation_sums_to_json_object_count(tmp_path):
    scene = _write_scene(
        tmp_path,
        [
            {"id": 0, "class_name": "a", "visible_pixels": 4},
            {"id": 1, "class_name": "b", "visible_pixels": 0},
            {"id": 2, "class_name": "c", "visible_pixels": 4},
        ],
        {
            0: [(2, 2), (2, 3), (3, 2), (3, 3)],
            2: [(10, 10), (10, 11), (11, 10), (11, 11)],
        },
    )
    ws = np.ones((40, 40), dtype=bool)
    ws[:, :2] = False
    gt = build_gt_scene(scene, ws, visibility=VisibilityFilter(mode="absolute", absolute_min=1))
    assert len(gt.gt_eval) + gt.gt_invisible + gt.gt_out_of_scope == gt.gt_total == 3


def test_relative_median_excludes_near_invisible(tmp_path):
    large_pixels = [(y, x) for y in range(5, 35) for x in range(5, 35)]
    scene = _write_scene(
        tmp_path,
        [
            {"id": 0, "class_name": "large", "visible_pixels": len(large_pixels)},
            {"id": 1, "class_name": "tiny", "visible_pixels": 2},
        ],
        {0: large_pixels, 1: [(1, 1), (1, 2)]},
    )
    ws = np.ones((40, 40), dtype=bool)
    gt_abs = build_gt_scene(scene, ws, visibility=VisibilityFilter(mode="absolute", absolute_min=1))
    gt_rel = build_gt_scene(
        scene,
        ws,
        visibility=VisibilityFilter(mode="relative_median", relative_fraction=0.01),
    )
    assert len(gt_abs.gt_eval) == 2
    assert gt_rel.visibility_threshold_px == 5  # ceil(0.01 * median([900, 2]))
    assert len(gt_rel.gt_eval) == 1
    assert gt_rel.gt_invisible == 1
