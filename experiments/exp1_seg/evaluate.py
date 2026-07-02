"""Phase B: offline evaluation of Experiment 1 predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.aggregate import Aggregator  # noqa: E402
from evaluation.gt import VisibilityFilter, build_gt_scene  # noqa: E402
from evaluation.matching import greedy_match  # noqa: E402
from evaluation.masks import decode_masks_rle, iou_matrix  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    StageMetrics,
    evaluate_boxes,
    evaluate_masks,
    metrics_to_dict,
)
from evaluation.reporting import write_reporting_tables  # noqa: E402
from evaluation.scene_registry import ALL_CATEGORY_ROWS, scene_meta  # noqa: E402
from experiments.exp1_seg.test_set import (  # noqa: E402
    filter_scene_id_files,
    list_test_set_names,
    test_set_manifest,
)


def _load_eval_config() -> dict:
    import yaml

    path = Path(__file__).parent / "eval_config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_preds(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    h, w = int(data["height"]), int(data["width"])
    ws = data["workspace_mask"].astype(bool)

    def _masks(key_rle: str) -> list[np.ndarray]:
        if key_rle not in data or data[key_rle].shape == ():
            return []
        return decode_masks_rle(data[key_rle], h, w)

    def _labels(key: str) -> list[str]:
        if key not in data:
            return []
        return [str(x) for x in data[key]]

    boxes = data["boxes_D"] if "boxes_D" in data else np.zeros((0, 4))
    return {
        "status": str(data.get("status", "ok")),
        "height": h,
        "width": w,
        "workspace_mask": ws,
        "boxes_D": [tuple(int(v) for v in b) for b in boxes],
        "scores_D": np.asarray(data.get("scores_D", []), dtype=np.float64),
        "masks_S": _masks("masks_S_rle"),
        "scores_S": np.asarray(data.get("scores_S", []), dtype=np.float64),
        "masks_M": _masks("masks_M_rle"),
        "scores_M": np.asarray(data.get("scores_M", []), dtype=np.float64),
        "masks_F": _masks("masks_F_rle"),
        "scores_F": np.asarray(data.get("scores_F", []), dtype=np.float64),
    }


def _class_recall_at_f(
    pred_masks: list[np.ndarray],
    scores: np.ndarray,
    gt_scene,
) -> dict[str, tuple[int, int]]:
    gt_by_class: dict[str, list] = {}
    for gi in gt_scene.gt_eval:
        gt_by_class.setdefault(gi.class_name, []).append(gi.mask)

    if not pred_masks:
        return {cls: (0, len(ms)) for cls, ms in gt_by_class.items()}

    out: dict[str, tuple[int, int]] = {}
    for cls, gt_masks in gt_by_class.items():
        ious = iou_matrix(pred_masks, gt_masks)
        sc = np.asarray(scores, dtype=np.float64) if len(scores) else np.zeros(len(pred_masks))
        match = greedy_match(ious, sc, 0.5)
        tp = len(match.tp_pairs)
        fn = len(match.fn_gt_indices)
        out[cls] = (tp, fn)
    return out


def _visibility_filter_from_args(args, eval_cfg: dict) -> VisibilityFilter:
    mode = args.visibility_mode or eval_cfg.get("visibility_mode", "absolute")
    if mode == "absolute":
        absolute_min = (
            args.min_visible
            if args.min_visible is not None
            else int(eval_cfg.get("min_visible_pixels_primary", 1))
        )
        return VisibilityFilter(mode="absolute", absolute_min=absolute_min)
    fraction = (
        args.visibility_relative_fraction
        if args.visibility_relative_fraction is not None
        else float(eval_cfg.get("visibility_relative_fraction", 0.01))
    )
    return VisibilityFilter(mode="relative_median", relative_fraction=fraction)


def _visibility_label(vf: VisibilityFilter) -> str:
    if vf.mode == "relative_median":
        return f"relative_median_{vf.relative_fraction:g}"
    return f"absolute_{vf.absolute_min}px"


def evaluate_scene(
    session_path: Path,
    preds: dict,
    eval_cfg: dict,
    *,
    clip_workspace: bool = True,
    visibility: VisibilityFilter | None = None,
    min_visible: int | None = None,
) -> dict:
    ws = preds["workspace_mask"]
    if visibility is None:
        visibility = (
            VisibilityFilter.from_legacy_min_visible(min_visible)
            if min_visible is not None
            else VisibilityFilter.from_legacy_min_visible(
                int(eval_cfg.get("min_visible_pixels_primary", 1))
            )
        )
    gt_scene = build_gt_scene(
        session_path,
        ws,
        visibility=visibility,
        workspace_majority_threshold=float(
            eval_cfg.get("workspace_majority_threshold", 0.5)
        ),
    )

    if not clip_workspace:
        gt_scene = build_gt_scene(
            session_path,
            np.ones_like(ws),
            visibility=visibility,
            workspace_majority_threshold=float(
                eval_cfg.get("workspace_majority_threshold", 0.5)
            ),
        )

    gt_masks = [g.mask for g in gt_scene.gt_eval]
    gt_boxes = [g.bbox for g in gt_scene.gt_eval if g.bbox is not None]

    small_max = int(eval_cfg.get("size_buckets_px2", {}).get("small_max", 900))
    medium_max = int(eval_cfg.get("size_buckets_px2", {}).get("medium_max", 6400))

    stage_metrics = {}
    stage_metrics["D"] = evaluate_boxes(
        preds["boxes_D"],
        preds["scores_D"],
        gt_boxes,
        point="D",
    )
    for pt in ("S", "M", "F"):
        stage_metrics[pt] = evaluate_masks(
            preds["masks_S"] if pt == "S" else preds["masks_M"] if pt == "M" else preds["masks_F"],
            preds["scores_S"] if pt == "S" else preds["scores_M"] if pt == "M" else preds["scores_F"],
            gt_masks,
            point=pt,
            small_max=small_max,
            medium_max=medium_max,
        )

    meta = scene_meta(session_path)
    class_recall = _class_recall_at_f(preds["masks_F"], preds["scores_F"], gt_scene)

    return {
        "scene_id": meta.scene_id,
        "category": meta.category,
        "viewpoint": meta.viewpoint,
        "status": preds["status"],
        "visibility": {
            "mode": visibility.mode,
            "absolute_min": visibility.absolute_min,
            "relative_fraction": visibility.relative_fraction,
            "threshold_px": gt_scene.visibility_threshold_px,
        },
        "gt_counts": {
            "eval": len(gt_scene.gt_eval),
            "invisible": gt_scene.gt_invisible,
            "out_of_scope": gt_scene.gt_out_of_scope,
            "total_instances": gt_scene.gt_total,
        },
        "stages": {k: metrics_to_dict(v) for k, v in stage_metrics.items()},
        "class_recall_F": {k: {"tp": v[0], "fn": v[1]} for k, v in class_recall.items()},
    }


def _build_summary(
    agg: Aggregator,
    eval_cfg: dict,
    *,
    region: str,
    test_set: str | None,
    visibility: VisibilityFilter,
) -> dict:
    category_order = eval_cfg.get("category_rows", list(ALL_CATEGORY_ROWS))
    failure_rows = []
    for row in agg.per_stage_rows():
        failure_rows.append({
            "point": row["label"],
            "category": "all",
            "over_segmentation_rate": row["over_segmentation_rate"],
            "merge_rate": row["merge_rate"],
            "hallucination_rate": row["hallucination_rate"],
            "miss_rate": row["miss_rate"],
        })
    for row in agg.per_category_F_rows(category_order):
        failure_rows.append({
            "point": "F",
            "category": row["label"],
            "over_segmentation_rate": row["over_segmentation_rate"],
            "merge_rate": row["merge_rate"],
            "hallucination_rate": row["hallucination_rate"],
            "miss_rate": row["miss_rate"],
        })

    return {
        "region": region,
        "test_set": test_set,
        "visibility": {
            "mode": visibility.mode,
            "absolute_min": visibility.absolute_min,
            "relative_fraction": visibility.relative_fraction,
            "label": _visibility_label(visibility),
        },
        "min_visible_pixels": visibility.absolute_min if visibility.mode == "absolute" else None,
        "gt_reconciliation": {
            "gt_eval": agg.gt_eval_total,
            "gt_invisible": agg.gt_invisible,
            "gt_out_of_scope": agg.gt_out_of_scope,
            "sum": agg.gt_eval_total + agg.gt_invisible + agg.gt_out_of_scope,
        },
        "per_stage": agg.per_stage_rows(),
        "per_category_F": agg.per_category_F_rows(category_order),
        "per_class_recall_F": agg.per_class_recall(),
        "failure_modes": failure_rows,
        "score_note": eval_cfg.get("score_note", "").strip(),
    }


def run_evaluation(
    run_dir: Path,
    eval_cfg: dict,
    *,
    visibility: VisibilityFilter,
    region: str = "workspace",
    test_set: str | None = None,
    data_root: str | None = None,
    write_outputs: bool = True,
    metrics_subdir: str | None = None,
    summary_name: str = "summary.json",
) -> tuple[dict, Aggregator, list[dict]]:
    if data_root:
        import config as cfg

        cfg.BASE_PATH = data_root

    from config import BASE_PATH

    preds_dir = run_dir / "preds"
    metrics_dir = run_dir / (metrics_subdir or "metrics")
    if write_outputs:
        metrics_dir.mkdir(parents=True, exist_ok=True)

    agg = Aggregator()
    scene_records: list[dict] = []

    npz_files = sorted(preds_dir.glob("scene_*.npz"))
    if test_set:
        npz_files = filter_scene_id_files(npz_files, test_set, eval_cfg)
        scene_ids = [p.stem for p in npz_files]
        if write_outputs:
            with open(run_dir / "test_set_eval.json", "w", encoding="utf-8") as f:
                json.dump(test_set_manifest(test_set, eval_cfg, scene_ids), f, indent=2)

    for npz_path in npz_files:
        scene_id = npz_path.stem
        session_path = Path(BASE_PATH) / scene_id
        preds = _load_preds(npz_path)
        record = evaluate_scene(
            session_path,
            preds,
            eval_cfg,
            clip_workspace=(region == "workspace"),
            visibility=visibility,
        )
        if write_outputs:
            out_path = metrics_dir / f"{scene_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        scene_records.append(record)

        stage_objs = {k: StageMetrics(**v) for k, v in record["stages"].items()}
        class_recall = {k: (v["tp"], v["fn"]) for k, v in record["class_recall_F"].items()}
        agg.add_scene(stage_objs, record["category"], class_recall, record["gt_counts"])

    summary = _build_summary(
        agg,
        eval_cfg,
        region=region,
        test_set=test_set,
        visibility=visibility,
    )
    if write_outputs:
        with open(run_dir / summary_name, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if summary_name == "summary.json":
            write_reporting_tables(run_dir, summary)
    return summary, agg, scene_records


def _comparison_row(point: str, primary: dict, strict: dict) -> dict:
    p = next(r for r in primary["per_stage"] if r["point"] == point)
    s = next(r for r in strict["per_stage"] if r["point"] == point)
    return {
        "point": point,
        "gt_eval_primary": p["n_gt_eval"],
        "gt_eval_strict": s["n_gt_eval"],
        "gt_eval_delta": s["n_gt_eval"] - p["n_gt_eval"],
        "recall_primary": p["recall"],
        "recall_strict": s["recall"],
        "precision_primary": p["precision"],
        "precision_strict": s["precision"],
        "f1_primary": p["f1"],
        "f1_strict": s["f1"],
        "pq_primary": p["pq"],
        "pq_strict": s["pq"],
        "ap_primary": p["ap_macro"],
        "ap_strict": s["ap_macro"],
        "mean_iou_primary": p["mean_iou"],
        "mean_iou_strict": s["mean_iou"],
    }


def _write_comparison_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _print_comparison_table(rows: list[dict]) -> None:
    header = (
        f"{'point':>5} | {'gt_eval':>7} {'gt_eval':>7} | "
        f"{'recall':>7} {'recall':>7} | {'prec':>7} {'prec':>7} | "
        f"{'F1':>7} {'F1':>7} | {'PQ':>7} {'PQ':>7} | {'AP':>7} {'AP':>7}"
    )
    print(header)
    print(
        f"{'':>5} | {'primary':>7} {'strict':>7} | "
        f"{'primary':>7} {'strict':>7} | "
        f"{'primary':>7} {'strict':>7} | "
        f"{'primary':>7} {'strict':>7} | "
        f"{'primary':>7} {'strict':>7} | "
        f"{'primary':>7} {'strict':>7}"
    )
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['point']:>5} | "
            f"{row['gt_eval_primary']:7d} {row['gt_eval_strict']:7d} | "
            f"{row['recall_primary']:7.4f} {row['recall_strict']:7.4f} | "
            f"{row['precision_primary']:7.4f} {row['precision_strict']:7.4f} | "
            f"{row['f1_primary']:7.4f} {row['f1_strict']:7.4f} | "
            f"{row['pq_primary']:7.4f} {row['pq_strict']:7.4f} | "
            f"{row['ap_primary']:7.4f} {row['ap_strict']:7.4f}"
        )


def _print_category_recall_comparison(primary: dict, strict: dict) -> None:
    p_map = {r["label"]: r["recall"] for r in primary["per_category_F"]}
    s_map = {r["label"]: r["recall"] for r in strict["per_category_F"]}
    cats = sorted(set(p_map) | set(s_map))
    print("\nPer-category recall @ F:")
    print(f"{'category':<18} {'primary':>8} {'strict':>8}")
    print("-" * 36)
    for cat in cats:
        print(f"{cat:<18} {p_map.get(cat, 0.0):8.4f} {s_map.get(cat, 0.0):8.4f}")


def run_threshold_comparison(
    run_dir: Path,
    eval_cfg: dict,
    *,
    region: str,
    test_set: str | None,
    data_root: str | None,
) -> dict:
    primary_vf = VisibilityFilter(
        mode="absolute",
        absolute_min=int(eval_cfg.get("min_visible_pixels_primary", 1)),
    )
    strict_vf = VisibilityFilter(
        mode="relative_median",
        relative_fraction=float(eval_cfg.get("visibility_relative_fraction", 0.01)),
    )

    print("[EXP1-EVAL] Threshold comparison — primary (absolute >= 1 px)")
    summary_primary, _, _ = run_evaluation(
        run_dir,
        eval_cfg,
        visibility=primary_vf,
        region=region,
        test_set=test_set,
        data_root=data_root,
        write_outputs=True,
        metrics_subdir="metrics_primary",
        summary_name="summary_primary.json",
    )

    print("[EXP1-EVAL] Threshold comparison — strict (relative_median 1%)")
    summary_strict, _, _ = run_evaluation(
        run_dir,
        eval_cfg,
        visibility=strict_vf,
        region=region,
        test_set=test_set,
        data_root=data_root,
        write_outputs=True,
        metrics_subdir="metrics_strict",
        summary_name="summary_strict.json",
    )

    comparison_rows = [
        _comparison_row(pt, summary_primary, summary_strict) for pt in ("D", "S", "M", "F")
    ]
    cmp_path = run_dir / "tables" / "threshold_comparison.csv"
    _write_comparison_csv(cmp_path, comparison_rows)

    moved_out = (
        summary_primary["gt_reconciliation"]["gt_eval"]
        - summary_strict["gt_reconciliation"]["gt_eval"]
    )
    comparison = {
        "primary": summary_primary["visibility"],
        "strict": summary_strict["visibility"],
        "gt_reconciliation_primary": summary_primary["gt_reconciliation"],
        "gt_reconciliation_strict": summary_strict["gt_reconciliation"],
        "instances_moved_out_of_denominator": moved_out,
        "per_stage_comparison": comparison_rows,
    }
    with open(run_dir / "threshold_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Keep canonical summary.json / §9.1 tables aligned with the primary threshold.
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_primary, f, indent=2)
    write_reporting_tables(run_dir, summary_primary)

    print(f"\n[EXP1-EVAL] GT reconciliation (primary): {summary_primary['gt_reconciliation']}")
    print(f"[EXP1-EVAL] GT reconciliation (strict):  {summary_strict['gt_reconciliation']}")
    print(f"[EXP1-EVAL] Instances moved out of denominator: {moved_out}")
    _print_comparison_table(comparison_rows)
    _print_category_recall_comparison(summary_primary, summary_strict)
    print(f"\n[EXP1-EVAL] Comparison CSV -> {cmp_path}")
    return comparison


def main() -> None:
    eval_cfg = _load_eval_config()
    test_names = list_test_set_names(eval_cfg)

    parser = argparse.ArgumentParser(description="Experiment 1 — Phase B offline evaluation")
    parser.add_argument("--run-dir", type=str, required=True, help="Results/exp1_seg/<timestamp>")
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        choices=test_names if test_names else None,
        metavar="NAME",
        help=f"Evaluate only scenes from named subset: {', '.join(test_names)}",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--region", choices=("workspace", "full"), default="workspace")
    parser.add_argument(
        "--min-visible",
        type=int,
        default=None,
        help="Absolute visibility minimum (overrides config; forces absolute mode unless --visibility-mode set)",
    )
    parser.add_argument(
        "--visibility-mode",
        choices=("absolute", "relative_median"),
        default=None,
        help="GT visibility filter mode (default: absolute)",
    )
    parser.add_argument(
        "--visibility-relative-fraction",
        type=float,
        default=None,
        help="For relative_median: keep instances with vis >= fraction * scene median (default 0.01)",
    )
    parser.add_argument(
        "--compare-thresholds",
        action="store_true",
        help="Run primary (1 px) and strict (relative_median 1%%) side by side; no Phase A",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    if args.compare_thresholds:
        run_threshold_comparison(
            run_dir,
            eval_cfg,
            region=args.region,
            test_set=args.test_set,
            data_root=args.data_root,
        )
        return

    visibility = _visibility_filter_from_args(args, eval_cfg)
    if args.min_visible is not None and args.visibility_mode is None:
        visibility = VisibilityFilter(mode="absolute", absolute_min=args.min_visible)

    npz_files = sorted((run_dir / "preds").glob("scene_*.npz"))
    if args.test_set:
        npz_files = filter_scene_id_files(npz_files, args.test_set, eval_cfg)
        print(f"[EXP1-EVAL] Test set '{args.test_set}': {len(npz_files)} prediction files")
    else:
        print(f"[EXP1-EVAL] {len(npz_files)} prediction files")
    print(f"[EXP1-EVAL] Visibility filter: {_visibility_label(visibility)}")

    summary, agg, _ = run_evaluation(
        run_dir,
        eval_cfg,
        visibility=visibility,
        region=args.region,
        test_set=args.test_set,
        data_root=args.data_root,
        write_outputs=True,
    )
    print(f"[EXP1-EVAL] GT sum: {summary['gt_reconciliation']['sum']}")
    print(f"[EXP1-EVAL] Summary -> {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
