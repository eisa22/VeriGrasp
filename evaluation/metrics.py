"""Segmentation metrics: AP, PQ, failure modes, box recall."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evaluation.matching import MatchResult, greedy_match
from evaluation.masks import box_iou, iou_matrix, mask_area


IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


@dataclass
class StageMetrics:
    point: str
    ap: float = 0.0
    ap50: float = 0.0
    ap75: float = 0.0
    ap_small: float = 0.0
    ap_medium: float = 0.0
    ap_large: float = 0.0
    mean_iou: float = 0.0
    pq: float = 0.0
    sq: float = 0.0
    rq: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_pred: int = 0
    n_gt_eval: int = 0
    over_segmentation_rate: float = 0.0
    merge_rate: float = 0.0
    hallucination_rate: float = 0.0
    miss_rate: float = 0.0
    box_recall: float | None = None


@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_iou_sum: float = 0.0
    n_gt: int = 0
    n_pred: int = 0
    over_seg: int = 0
    merge: int = 0
    hallucination: int = 0
    miss: int = 0
    ap_per_scene: list[float] = field(default_factory=list)
    ap50_per_scene: list[float] = field(default_factory=list)
    ap75_per_scene: list[float] = field(default_factory=list)


def _interp_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point COCO interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap_at_threshold(
    ious: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> float:
    n_gt = ious.shape[1]
    if n_gt == 0:
        return 0.0
    if ious.shape[0] == 0:
        return 0.0

    order = np.argsort(-scores)
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))
    gt_matched = np.zeros(n_gt, dtype=bool)

    for k, p in enumerate(order):
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
            tp[k] = 1
        else:
            fp[k] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return _interp_ap(recalls, precisions)


def compute_map(ious: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    aps = [compute_ap_at_threshold(ious, scores, float(t)) for t in IOU_THRESHOLDS]
    ap = float(np.mean(aps)) if aps else 0.0
    ap50 = compute_ap_at_threshold(ious, scores, 0.5)
    ap75 = compute_ap_at_threshold(ious, scores, 0.75)
    return ap, ap50, ap75


def _pq_from_match(match: MatchResult, ious: np.ndarray) -> tuple[float, float, float, float]:
    tp = len(match.tp_pairs)
    fp = len(match.fp_pred_indices)
    fn = len(match.fn_gt_indices)
    if tp == 0:
        return 0.0, 0.0, 0.0, 0.0
    sq = float(np.mean([ious[p, g] for p, g, _ in match.tp_pairs]))
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = float(tp / denom) if denom > 0 else 0.0
    pq = sq * rq
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return pq, sq, rq, prec if rec == 0 else 2 * prec * rec / (prec + rec)


def failure_modes(ious: np.ndarray, n_gt: int, n_pred: int) -> tuple[float, float, float, float]:
    if n_gt == 0:
        over = 0.0
        miss = 0.0
    else:
        over = float(np.sum(np.sum(ious >= 0.10, axis=0) >= 2) / n_gt)
        miss = float(np.sum(np.max(ious, axis=0) < 0.50) / n_gt) if n_pred > 0 else 1.0
    if n_pred == 0:
        merge = 0.0
        hall = 0.0
    else:
        merge = float(np.sum(np.sum(ious >= 0.10, axis=1) >= 2) / n_pred)
        hall = float(np.sum(np.max(ious, axis=1) < 0.10) / n_pred)
    return over, merge, hall, miss


def size_bucket(area: int, small_max: int, medium_max: int) -> str:
    if area < small_max:
        return "small"
    if area <= medium_max:
        return "medium"
    return "large"


def evaluate_masks(
    pred_masks: list[np.ndarray],
    scores: np.ndarray,
    gt_masks: list[np.ndarray],
    *,
    point: str = "F",
    small_max: int = 900,
    medium_max: int = 6400,
) -> StageMetrics:
    ious = iou_matrix(pred_masks, gt_masks)
    n_pred, n_gt = ious.shape
    sc = np.asarray(scores, dtype=np.float64) if len(scores) else np.zeros(n_pred)

    if n_pred == 0 and n_gt == 0:
        return StageMetrics(point=point)

    if n_pred == 0:
        over, merge, hall, miss = failure_modes(ious, n_gt, 0)
        return StageMetrics(
            point=point,
            fn=n_gt,
            n_gt_eval=n_gt,
            miss_rate=miss,
        )

    if n_gt == 0:
        over, merge, hall, miss = failure_modes(ious, 0, n_pred)
        return StageMetrics(
            point=point,
            fp=n_pred,
            n_pred=n_pred,
            hallucination_rate=1.0,
        )

    ap, ap50, ap75 = compute_map(ious, sc)
    match = greedy_match(ious, sc, 0.5)
    tp = len(match.tp_pairs)
    fp = len(match.fp_pred_indices)
    fn = len(match.fn_gt_indices)
    mean_iou = float(np.mean([iou for _, _, iou in match.tp_pairs])) if tp else 0.0
    pq, sq, rq, f1 = _pq_from_match(match, ious)
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    over, merge, hall, miss = failure_modes(ious, n_gt, n_pred)

    gt_areas = [mask_area(g) for g in gt_masks]
    bucket_gt: dict[str, list[int]] = {"small": [], "medium": [], "large": []}
    for i, a in enumerate(gt_areas):
        bucket_gt[size_bucket(a, small_max, medium_max)].append(i)

    ap_by_size = {}
    for name, indices in bucket_gt.items():
        if not indices:
            ap_by_size[name] = 0.0
            continue
        sub_gt = [gt_masks[i] for i in indices]
        sub_ious = iou_matrix(pred_masks, sub_gt)
        ap_by_size[name] = compute_ap_at_threshold(sub_ious, sc, 0.5)

    return StageMetrics(
        point=point,
        ap=ap,
        ap50=ap50,
        ap75=ap75,
        ap_small=ap_by_size.get("small", 0.0),
        ap_medium=ap_by_size.get("medium", 0.0),
        ap_large=ap_by_size.get("large", 0.0),
        mean_iou=mean_iou,
        pq=pq,
        sq=sq,
        rq=rq,
        precision=prec,
        recall=rec,
        f1=2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
        tp=tp,
        fp=fp,
        fn=fn,
        n_pred=n_pred,
        n_gt_eval=n_gt,
        over_segmentation_rate=over,
        merge_rate=merge,
        hallucination_rate=hall,
        miss_rate=miss,
    )


def evaluate_boxes(
    pred_boxes: list[tuple[int, int, int, int]],
    scores: np.ndarray,
    gt_boxes: list[tuple[int, int, int, int]],
    *,
    point: str = "D",
    iou_threshold: float = 0.5,
) -> StageMetrics:
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return StageMetrics(point=point, n_pred=n_pred, box_recall=0.0 if n_pred else 0.0)
    if n_pred == 0:
        return StageMetrics(point=point, fn=n_gt, n_gt_eval=n_gt, miss_rate=1.0, box_recall=0.0)

    ious = np.zeros((n_pred, n_gt), dtype=np.float64)
    for p, pb in enumerate(pred_boxes):
        for g, gb in enumerate(gt_boxes):
            ious[p, g] = box_iou(pb, gb)

    sc = np.asarray(scores, dtype=np.float64)
    match = greedy_match(ious, sc, iou_threshold)
    tp = len(match.tp_pairs)
    recall = float(tp / n_gt)
    over, merge, hall, miss = failure_modes(ious, n_gt, n_pred)

    return StageMetrics(
        point=point,
        recall=recall,
        box_recall=recall,
        tp=tp,
        fp=len(match.fp_pred_indices),
        fn=len(match.fn_gt_indices),
        n_pred=n_pred,
        n_gt_eval=n_gt,
        over_segmentation_rate=over,
        merge_rate=merge,
        hallucination_rate=hall,
        miss_rate=miss,
    )


def metrics_to_dict(m: StageMetrics) -> dict:
    return {k: getattr(m, k) for k in m.__dataclass_fields__}
