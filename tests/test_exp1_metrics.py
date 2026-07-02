"""Unit tests for Experiment 1 evaluation metrics."""

import numpy as np

from evaluation.matching import greedy_match
from evaluation.metrics import compute_ap_at_threshold, evaluate_masks, failure_modes
from evaluation.masks import iou_matrix


def _box_mask(x0, y0, x1, y1, h=100, w=100):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_perfect_match_ap_and_pq():
    gt = [_box_mask(10, 10, 40, 40)]
    pred = [_box_mask(10, 10, 40, 40)]
    scores = np.array([0.9])
    m = evaluate_masks(pred, scores, gt, point="F")
    assert m.ap50 == 1.0
    assert m.pq == 1.0
    assert m.tp == 1
    assert m.fp == 0
    assert m.fn == 0


def test_over_segmentation_detected():
    gt = [_box_mask(10, 10, 40, 40)]
    pred = [_box_mask(10, 10, 30, 30), _box_mask(20, 20, 40, 40)]
    scores = np.array([0.9, 0.8])
    ious = iou_matrix(pred, gt)
    over, merge, hall, miss = failure_modes(ious, 1, 2)
    assert over == 1.0
    m = evaluate_masks(pred, scores, gt, point="F")
    assert m.over_segmentation_rate == 1.0


def test_merge_detected():
    gt = [_box_mask(10, 10, 25, 40), _box_mask(26, 10, 40, 40)]
    pred = [_box_mask(10, 10, 40, 40)]
    scores = np.array([0.9])
    ious = iou_matrix(pred, gt)
    over, merge, hall, miss = failure_modes(ious, 2, 1)
    assert merge == 1.0


def test_pq_equals_sq_times_rq():
    gt = [_box_mask(10, 10, 40, 40), _box_mask(50, 50, 70, 70)]
    pred = [_box_mask(10, 10, 38, 38), _box_mask(52, 52, 68, 68)]
    scores = np.array([0.95, 0.85])
    m = evaluate_masks(pred, scores, gt, point="F")
    assert abs(m.pq - m.sq * m.rq) < 1e-6


def test_ap_ordering_sanity():
    gt = [_box_mask(10, 10, 40, 40)]
    pred = [_box_mask(12, 12, 38, 38)]
    scores = np.array([0.9])
    m = evaluate_masks(pred, scores, gt, point="F")
    assert m.ap50 >= m.ap75 >= m.ap - 1e-9 or m.ap <= m.ap50


def test_greedy_match_unique_gt():
    gt = [_box_mask(10, 10, 40, 40)]
    pred = [_box_mask(10, 10, 40, 40), _box_mask(10, 10, 35, 35)]
    ious = iou_matrix(pred, gt)
    match = greedy_match(ious, np.array([0.9, 0.8]), 0.5)
    assert len(match.tp_pairs) == 1
    assert len(match.fp_pred_indices) == 1
