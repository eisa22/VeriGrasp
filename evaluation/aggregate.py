"""Aggregate per-scene metrics into summary tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from evaluation.metrics import StageMetrics


@dataclass
class AggregateBucket:
    n_scenes: int = 0
    n_gt_eval: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_iou_sum: float = 0.0
    over_seg: int = 0
    merge: int = 0
    hallucination: int = 0
    miss: int = 0
    ap_values: list[float] = field(default_factory=list)
    ap50_values: list[float] = field(default_factory=list)
    ap75_values: list[float] = field(default_factory=list)
    box_recall_values: list[float] = field(default_factory=list)


def _micro_pq_r(bucket: AggregateBucket) -> tuple[float, float, float, float, float]:
    tp, fp, fn = bucket.tp, bucket.fp, bucket.fn
    if tp == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    sq = bucket.tp_iou_sum / tp
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom > 0 else 0.0
    pq = sq * rq
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return pq, sq, rq, prec, rec if f1 == 0 else f1


def add_stage(bucket: AggregateBucket, m: StageMetrics) -> None:
    bucket.n_scenes += 1
    bucket.n_gt_eval += m.n_gt_eval
    bucket.tp += m.tp
    bucket.fp += m.fp
    bucket.fn += m.fn
    bucket.tp_iou_sum += m.mean_iou * m.tp
    bucket.over_seg += int(round(m.over_segmentation_rate * m.n_gt_eval)) if m.n_gt_eval else 0
    bucket.merge += int(round(m.merge_rate * m.n_pred)) if m.n_pred else 0
    bucket.hallucination += int(round(m.hallucination_rate * m.n_pred)) if m.n_pred else 0
    bucket.miss += int(round(m.miss_rate * m.n_gt_eval)) if m.n_gt_eval else 0
    bucket.ap_values.append(m.ap)
    bucket.ap50_values.append(m.ap50)
    bucket.ap75_values.append(m.ap75)
    if m.box_recall is not None:
        bucket.box_recall_values.append(m.box_recall)


def bucket_summary(bucket: AggregateBucket, point: str, label: str) -> dict:
    pq, sq, rq, prec, f1 = _micro_pq_r(bucket)
    n_gt = bucket.n_gt_eval
    n_pred = bucket.tp + bucket.fp
    return {
        "label": label,
        "point": point,
        "n_scenes": bucket.n_scenes,
        "n_gt_eval": n_gt,
        "obj_per_scene": round(n_gt / bucket.n_scenes, 2) if bucket.n_scenes else 0.0,
        "ap_macro": float(np.mean(bucket.ap_values)) if bucket.ap_values else 0.0,
        "ap50_macro": float(np.mean(bucket.ap50_values)) if bucket.ap50_values else 0.0,
        "ap75_macro": float(np.mean(bucket.ap75_values)) if bucket.ap75_values else 0.0,
        "ap_micro": float(np.mean(bucket.ap_values)) if bucket.ap_values else 0.0,
        "mean_iou": bucket.tp_iou_sum / bucket.tp if bucket.tp else 0.0,
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "precision": prec,
        "recall": bucket.tp / (bucket.tp + bucket.fn) if (bucket.tp + bucket.fn) else 0.0,
        "f1": f1,
        "box_recall": float(np.mean(bucket.box_recall_values)) if bucket.box_recall_values else None,
        "tp": bucket.tp,
        "fp": bucket.fp,
        "fn": bucket.fn,
        "over_segmentation_rate": bucket.over_seg / n_gt if n_gt else 0.0,
        "merge_rate": bucket.merge / n_pred if n_pred else 0.0,
        "hallucination_rate": bucket.hallucination / n_pred if n_pred else 0.0,
        "miss_rate": bucket.miss / n_gt if n_gt else 0.0,
    }


class Aggregator:
    def __init__(self) -> None:
        self.by_stage: dict[str, AggregateBucket] = defaultdict(AggregateBucket)
        self.by_category_F: dict[str, AggregateBucket] = defaultdict(AggregateBucket)
        self.class_tp: dict[str, int] = defaultdict(int)
        self.class_fn: dict[str, int] = defaultdict(int)
        self.gt_invisible = 0
        self.gt_out_of_scope = 0
        self.gt_eval_total = 0

    def add_scene(
        self,
        scene_metrics: dict[str, StageMetrics],
        category: str,
        class_recall: dict[str, tuple[int, int]],
        gt_counts: dict[str, int],
    ) -> None:
        self.gt_invisible += gt_counts.get("invisible", 0)
        self.gt_out_of_scope += gt_counts.get("out_of_scope", 0)
        self.gt_eval_total += gt_counts.get("eval", 0)
        for point, m in scene_metrics.items():
            add_stage(self.by_stage[point], m)
            if point == "F":
                add_stage(self.by_category_F[category], m)
        for cls, (tp, fn) in class_recall.items():
            self.class_tp[cls] += tp
            self.class_fn[cls] += fn

    def per_stage_rows(self) -> list[dict]:
        rows = []
        for point in ("D", "S", "M", "F"):
            if point in self.by_stage:
                rows.append(bucket_summary(self.by_stage[point], point, point))
        return rows

    def per_category_F_rows(self, category_order: list[str]) -> list[dict]:
        rows = []
        for cat in category_order:
            b = self.by_category_F.get(cat)
            if b and b.n_scenes:
                rows.append(bucket_summary(b, "F", cat))
        return rows

    def per_class_recall(self) -> list[dict]:
        rows = []
        for cls in sorted(self.class_tp.keys() | self.class_fn.keys()):
            tp = self.class_tp.get(cls, 0)
            fn = self.class_fn.get(cls, 0)
            total = tp + fn
            rows.append({
                "class_name": cls,
                "tp": tp,
                "fn": fn,
                "recall": tp / total if total else 0.0,
            })
        return rows
