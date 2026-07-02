"""Greedy IoU matching (COCO-style and fixed-threshold)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    tp_pairs: list[tuple[int, int, float]]  # pred_idx, gt_idx, iou
    fp_pred_indices: list[int]
    fn_gt_indices: list[int]


def greedy_match(
    ious: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> MatchResult:
    """COCO greedy match: sort preds by score desc, assign to best unmatched GT."""
    n_pred, n_gt = ious.shape
    if n_pred == 0:
        return MatchResult([], [], list(range(n_gt)))
    if n_gt == 0:
        return MatchResult([], list(range(n_pred)), [])

    order = np.argsort(-scores)
    gt_matched = np.zeros(n_gt, dtype=bool)
    tp_pairs: list[tuple[int, int, float]] = []
    fp: list[int] = []

    for p in order:
        best_g = -1
        best_iou = iou_threshold
        for g in range(n_gt):
            if gt_matched[g]:
                continue
            iou = float(ious[p, g])
            if iou >= best_iou:
                best_iou = iou
                best_g = g
        if best_g >= 0:
            gt_matched[best_g] = True
            tp_pairs.append((int(p), int(best_g), best_iou))
        else:
            fp.append(int(p))

    fn = [g for g in range(n_gt) if not gt_matched[g]]
    return MatchResult(tp_pairs, fp, fn)


def match_at_threshold(ious: np.ndarray, scores: np.ndarray, iou_threshold: float) -> MatchResult:
    return greedy_match(ious, scores, iou_threshold)
