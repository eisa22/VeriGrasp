"""Experiment 3 — offline evaluation of verification layer effectiveness.

Reads persisted pipeline outputs (stages 0, 8, 10, 11, 12b, 13) plus dataset
ground truth. Re-runs verification in full mode when cascade audits are
incomplete. Never re-runs the detector.

Usage:
    python -m experiments.exp3_verification.evaluate \
        --data-root Data/blender_dataset \
        --out-dir Results/exp3/<run> \
        [--visibility strict|primary|both] [--limit N] [--secondary-sample]
"""

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

from evaluation.env_snapshot import _git_commit, config_hash  # noqa: E402
from evaluation.exp2_geometry import derive_pred_geometry  # noqa: E402
from evaluation.exp2_gt import load_gt_geometries  # noqa: E402
from evaluation.exp3_aggregate import build_summary  # noqa: E402
from evaluation.exp3_metrics import check_metrics, roc_curve  # noqa: E402
from evaluation.exp3_offline_verify import (  # noqa: E402
    CHECK_ORDER,
    compute_soft_score_from_checks,
    resolve_primary_grasp_json,
    run_full_verification,
    verification_row_fields,
)
from evaluation.exp3_oracle import (  # noqa: E402
    evaluate_oracle,
    load_gt_corner_boxes,
    oracle_params_from_config,
)
from evaluation.gt import VisibilityFilter, build_gt_scene  # noqa: E402
from evaluation.masks import clip_mask, decode_masks_rle, iou_matrix  # noqa: E402
from evaluation.matching import greedy_match  # noqa: E402
from evaluation.scene_registry import scene_meta  # noqa: E402

BASE_COLUMNS = [
    "scene_id",
    "category_band",
    "target_matched",
    "gt_class",
    "valid",
    "violated_criteria",
    "verdict_cascade",
    "decisive_check",
    "soft_score",
]


def _load_eval_config() -> dict:
    import yaml

    path = Path(__file__).parent / "eval_config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _decode_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    arr = np.array([np.asarray(counts, dtype=np.int32)], dtype=object)
    return decode_masks_rle(arr, height, width)[0]


def _csv_columns() -> list[str]:
    cols = list(BASE_COLUMNS)
    for name in CHECK_ORDER:
        cols.extend([
            f"check_{name}_pass",
            f"check_{name}_margin",
            f"check_{name}_unverifiable",
        ])
    return cols


def evaluate_scene(
    session_path: Path,
    eval_cfg: dict,
    visibility: VisibilityFilter,
    *,
    skip_verify_rerun: bool = False,
    secondary_sample: bool = False,
) -> dict:
    """Evaluate one scene; returns status counters and optional rows."""
    scene_id = session_path.name
    meta = scene_meta(session_path)
    band = meta.category
    oracle_params = oracle_params_from_config(eval_cfg)

    prep = _load_json(session_path / "stage_prep_context.json")
    if prep is None:
        return {
            "status": "missing_pipeline_output",
            "no_target": False,
            "grasp_row": None,
            "secondary_rows": [],
        }

    plane = tuple(float(x) for x in prep["plane_model"])
    h, w = int(prep["height"]), int(prep["width"])
    ws = _decode_rle(prep["workspace_mask_rle"], h, w).astype(bool)

    gt_scene = build_gt_scene(
        session_path,
        ws,
        visibility=visibility,
        workspace_majority_threshold=float(
            eval_cfg.get("workspace_majority_threshold", 0.5)
        ),
    )
    gt_geos = load_gt_geometries(session_path, plane)
    all_corners = load_gt_corner_boxes(session_path)

    stage8 = _load_json(session_path / "stage8_candidates.json")
    records = stage8["candidates"] if stage8 else []
    pred_masks = [_decode_rle(r["mask_rle"], h, w) for r in records]
    stage10 = _load_json(session_path / "stage10_selected_target.json")
    stage11 = _load_json(session_path / "stage11_suction_grasps.json")
    corridor = _load_json(session_path / "extraction_corridor.json")
    persisted_v = _load_json(session_path / "verification_result.json")

    primary_id = None
    if stage10 and stage10.get("primary"):
        primary_id = stage10["primary"]["candidate"]["candidate_id"]

    if primary_id is None:
        return {
            "status": "no_target",
            "no_target": True,
            "grasp_row": None,
            "secondary_rows": [],
        }

    pred_clipped = [clip_mask(m, ws) for m in pred_masks]
    scores = np.asarray([float(r.get("score", 0.0)) for r in records], dtype=np.float64)
    gt_masks = [g.mask for g in gt_scene.gt_eval]
    ious = iou_matrix(pred_clipped, gt_masks)
    match = greedy_match(ious, scores, float(eval_cfg.get("iou_threshold", 0.5)))

    primary_pred_idx = next(
        (i for i, r in enumerate(records) if r["candidate_id"] == primary_id),
        None,
    )
    if primary_pred_idx is None:
        return {
            "status": "stale_target",
            "no_target": False,
            "grasp_row": None,
            "secondary_rows": [],
        }

    matched_gt_by_pred = {p: (g, iou) for p, g, iou in match.tp_pairs}
    target_matched = primary_pred_idx in matched_gt_by_pred

    candidate_record = records[primary_pred_idx]
    primary_grasp = resolve_primary_grasp_json(
        stage10,
        stage11,
        candidate_record,
        allow_centroid_fallback=not target_matched,
    )

    if not primary_grasp:
        return {
            "status": "no_grasp",
            "no_target": False,
            "grasp_row": None,
            "secondary_rows": [],
        }

    matched_gt = None
    matched_corners = None
    gt_class = ""
    pred_yaw = None

    if target_matched:
        g_idx, _ = matched_gt_by_pred[primary_pred_idx]
        gi = gt_scene.gt_eval[g_idx]
        matched_gt = gt_geos.get(gi.instance_id)
        matched_corners = all_corners.get(gi.instance_id)
        gt_class = matched_gt.class_name if matched_gt else gi.class_name
        pred_geo = derive_pred_geometry(records[primary_pred_idx], plane)
        pred_yaw = pred_geo.footprint_yaw_deg

    label = evaluate_oracle(
        grasp_position=np.asarray(primary_grasp["position"], dtype=np.float64),
        grasp_normal=np.asarray(primary_grasp.get("normal", [0, 0, -1]), dtype=np.float64),
        pred_yaw_deg=pred_yaw,
        target_matched=target_matched,
        matched_gt=matched_gt,
        matched_corners=matched_corners,
        all_gt_geos=gt_geos,
        all_corners=all_corners,
        plane=plane,
        params=oracle_params,
    )

    candidate_record = records[primary_pred_idx]
    verification = run_full_verification(
        session_path,
        prep,
        candidate_record,
        primary_grasp,
        corridor,
        skip_rerun=skip_verify_rerun,
        persisted=persisted_v,
    )

    soft_score = compute_soft_score_from_checks(verification.checks)

    row = {
        "scene_id": scene_id,
        "category_band": band,
        "target_matched": target_matched,
        "gt_class": gt_class,
        "valid": label.valid,
        "violated_criteria": ";".join(label.violated),
        "verdict_cascade": verification.verdict_cascade,
        "decisive_check": verification.decisive_check or "",
        "soft_score": soft_score,
    }
    row.update(verification_row_fields(verification.checks))

    secondary_rows: list[dict] = []
    if secondary_sample and stage11:
        for grasp_json in stage11.get("grasps") or []:
            sec = run_full_verification(
                session_path,
                prep,
                candidate_record,
                grasp_json,
                corridor,
                skip_rerun=False,
                persisted=None,
            )
            secondary_rows.append({
                "scene_id": scene_id,
                "grasp_rank": grasp_json.get("rank"),
                **verification_row_fields(sec.checks),
            })

    return {
        "status": "ok",
        "no_target": False,
        "grasp_row": row,
        "secondary_rows": secondary_rows,
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


def _write_roc_figure(
    grasp_rows: list[dict],
    summary: dict,
    fig_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = np.asarray([bool(r["valid"]) for r in grasp_rows], dtype=bool)
    scores = np.asarray([float(r["soft_score"]) for r in grasp_rows], dtype=np.float64)
    roc = summary["roc_soft_score"]
    cascade_pt = roc.get("cascade_point", {})

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(roc["fpr"], roc["tpr"], color="tab:blue", linewidth=1.5, label="soft score")
    ax.scatter(
        [cascade_pt.get("fpr", 0.0)],
        [cascade_pt.get("tpr", 0.0)],
        color="tab:red",
        s=40,
        zorder=5,
        label="cascade",
    )

    # Thin grey per-check ROCs when readable.
    for name in CHECK_ORDER:
        margins = []
        labels = []
        for r in grasp_rows:
            uv = r.get(f"check_{name}_unverifiable")
            m = r.get(f"check_{name}_margin")
            if uv is True or uv == "True" or m == "" or m is None:
                continue
            margins.append(float(m))
            labels.append(bool(r["valid"]))
        if len(margins) < 10:
            continue
        margin_arr = np.asarray(margins, dtype=np.float64)
        if np.allclose(margin_arr, margin_arr[0]):
            continue
        chk_roc = roc_curve(margin_arr, np.asarray(labels, dtype=bool), higher_is_pass=True)
        ax.plot(
            chk_roc["fpr"],
            chk_roc["tpr"],
            color="0.75",
            linewidth=0.4,
            alpha=0.5,
        )

    ax.plot([0, 1], [0, 1], color="0.85", linestyle="--", linewidth=0.8)
    ax.set_xlabel("FPR (FAR)")
    ax.set_ylabel("TPR (1 − FRR)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[FIGURE] geschrieben -> {fig_path}")


def run_evaluation(
    session_paths: list[Path],
    eval_cfg: dict,
    visibility: VisibilityFilter,
    out_dir: Path,
    *,
    label: str,
    run_id: str | None = None,
    skip_verify_rerun: bool = False,
    secondary_sample: bool = False,
    fig_root: Path | None = None,
) -> dict:
    grasp_rows: list[dict] = []
    secondary_all: list[dict] = []
    n_no_target = 0
    n_missing = 0
    n_no_grasp = 0
    n_stale_target = 0
    n_target_unmatched = 0

    for i, sp in enumerate(session_paths, start=1):
        result = evaluate_scene(
            sp,
            eval_cfg,
            visibility,
            skip_verify_rerun=skip_verify_rerun,
            secondary_sample=secondary_sample,
        )
        if result["status"] == "missing_pipeline_output":
            n_missing += 1
            continue
        if result["no_target"]:
            n_no_target += 1
            continue
        if result["status"] == "no_grasp":
            n_no_grasp += 1
            continue
        if result["status"] == "stale_target":
            n_stale_target += 1
            continue
        row = result.get("grasp_row")
        if row is None:
            continue
        if not row["target_matched"]:
            n_target_unmatched += 1
        grasp_rows.append(row)
        secondary_all.extend(result.get("secondary_rows") or [])
        if i % 50 == 0 or i == len(session_paths):
            print(f"[{label}] {i}/{len(session_paths)} Szenen — {len(grasp_rows)} Grasp-Zeilen")

    band_order = list(eval_cfg.get("category_rows", []))
    meta = {
        "run_id": run_id or out_dir.name,
        "config_hash": config_hash(eval_cfg),
        "git_commit": _git_commit(),
        "visibility_rule": (
            "relative_median" if visibility.mode == "relative_median" else "primary"
        ),
        "n_scenes": len(session_paths),
        "n_no_target": n_no_target,
        "n_no_grasp": n_no_grasp,
        "n_stale_target": n_stale_target,
        "n_grasps": len(grasp_rows),
        "n_target_unmatched": n_target_unmatched,
        "n_scenes_missing_pipeline_output": n_missing,
        "oracle_params": {
            "tol_contain_mm": int(eval_cfg.get("tol_contain_mm", 5)),
            "tol_contact_mm": int(eval_cfg.get("tol_contact_mm", 20)),
            "tol_normal_deg": int(eval_cfg.get("tol_normal_deg", 15)),
        },
        "open_points": {
            "A": "Canonical run persists cascade only; full-mode margins recomputed offline.",
            "B": "All check margins: higher_is_pass (verification.types convention).",
            "C": "Secondary sample supported offline via verify_grasp per ranked grasp.",
        },
    }
    summary = build_summary(grasp_rows, band_order, meta)

    if secondary_sample and secondary_all:
        sec_checks = [check_metrics(secondary_all, name) for name in CHECK_ORDER]
        summary["roc_checks_secondary"] = sec_checks

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = _csv_columns()
    _write_csv(out_dir / "exp3_per_grasp.csv", grasp_rows, columns)
    with open(out_dir / "exp3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[{label}] Summary -> {out_dir / 'exp3_summary.json'}")

    if fig_root is not None and grasp_rows:
        _write_roc_figure(
            grasp_rows,
            summary,
            fig_root / "figures" / "exp3_roc.pdf",
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 — offline evaluation")
    parser.add_argument(
        "--data-root", type=Path,
        default=PROJECT_ROOT / "Data" / "blender_dataset",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--visibility", choices=["primary", "strict", "both"], default="strict",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument(
        "--secondary-sample",
        action="store_true",
        help="Also run full verification on all ranked grasps (not in thesis tables)",
    )
    parser.add_argument(
        "--skip-verify-rerun",
        action="store_true",
        help="Debug: reuse persisted verification when complete (not for production)",
    )
    args = parser.parse_args()

    eval_cfg = _load_eval_config()
    sessions = sorted(
        p for p in args.data_root.iterdir()
        if p.is_dir() and p.name.startswith("scene_")
    )
    if args.scenes:
        wanted = set(args.scenes.split(","))
        sessions = [p for p in sessions if p.name in wanted]
    if args.limit is not None:
        sessions = sessions[: args.limit]
    if not sessions:
        raise SystemExit(f"Keine Szenen gefunden unter {args.data_root}")
    print(f"[EXP3] {len(sessions)} Szenen aus {args.data_root}")

    if args.out_dir is None:
        from datetime import datetime

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.out_dir = PROJECT_ROOT / "Results" / "exp3" / stamp

    filters = {
        "primary": VisibilityFilter(
            mode="absolute",
            absolute_min=int(eval_cfg.get("min_visible_pixels_primary", 1)),
        ),
        "strict": VisibilityFilter(
            mode="relative_median",
            relative_fraction=float(eval_cfg.get("visibility_relative_fraction", 0.01)),
        ),
    }
    selected = ["primary", "strict"] if args.visibility == "both" else [args.visibility]

    for i, name in enumerate(selected):
        sub_dir = args.out_dir / name if len(selected) > 1 else args.out_dir
        run_evaluation(
            sessions,
            eval_cfg,
            filters[name],
            sub_dir,
            label=name,
            run_id=args.out_dir.name,
            skip_verify_rerun=args.skip_verify_rerun,
            secondary_sample=args.secondary_sample,
            fig_root=args.out_dir if i == 0 else None,
        )


if __name__ == "__main__":
    main()
