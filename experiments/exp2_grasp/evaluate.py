"""Experiment 2 — offline evaluation of centroid / normal estimation accuracy.

Reads the persisted per-stage JSON outputs of an existing pipeline run
(``stage_prep_context.json``, ``stage8_candidates.json``,
``stage10_selected_target.json``, ``stage11_suction_grasps.json``) plus the
dataset ground truth. It never re-runs the pipeline or the detector.

Matching reuses the Experiment 1 matcher (unique greedy, IoU >= 0.5, inside
the workspace mask, same evaluability rules).

Usage:
    python -m experiments.exp2_grasp.evaluate \
        --data-root Data/blender_dataset \
        --out-dir Results/exp2/<run> \
        [--visibility primary|strict|both] [--limit N] [--gt-self-test]
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
from evaluation.exp2_aggregate import build_summary  # noqa: E402
from evaluation.exp2_geometry import (  # noqa: E402
    angle_between_deg,
    derive_pred_geometry,
    orient_toward_camera,
)
from evaluation.exp2_gt import (  # noqa: E402
    GtObjectGeometry,
    load_gt_geometries,
    visibility_ratio,
)
from evaluation.exp2_metrics import candidate_errors  # noqa: E402
from evaluation.gt import VisibilityFilter, build_gt_scene  # noqa: E402
from evaluation.masks import clip_mask, decode_masks_rle, iou_matrix  # noqa: E402
from evaluation.matching import greedy_match  # noqa: E402
from evaluation.scene_registry import scene_meta  # noqa: E402
from perception.geometry.plane import plane_basis  # noqa: E402

PER_CANDIDATE_COLUMNS = [
    "scene_id", "category_band", "gt_instance_id", "gt_class", "packaging_type",
    "visible_px", "visibility_ratio", "match_iou",
    "e_lat_mm", "e_top_mm_signed",
    "ext_err_long_rel", "ext_err_short_rel", "ext_err_height_rel",
    "yaw_err_deg", "yaw_fold_deg",
    "e_bottom_mm_signed", "bottom_cue", "bottom_confidence",
    "is_primary_target",
]

PER_GRASP_COLUMNS = [
    "scene_id", "category_band", "gt_class", "theta_deg",
    "within_12deg", "within_30deg", "status",
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


def _is_degenerate_plane_fit(prep: dict) -> bool:
    """Detect the RANSAC fallback plane ([0,0,1,0]: a plane through the camera).

    A healthy pallet plane lies at roughly z_pallet_m (~2-3 m) from the camera
    origin; the degenerate fallback passes (almost) through the origin. In that
    broken frame the pipeline's from_pallet snap (bottom_z := 0) lands metres
    above the parcel — a genuine pipeline error, reported separately so it
    does not distort the regular fallback_pallet cue row.
    """
    a, b, c, d = (float(x) for x in prep["plane_model"])
    norm = (a * a + b * b + c * c) ** 0.5 + 1e-12
    origin_dist = abs(d) / norm
    return origin_dist < 0.5 * float(prep["z_pallet_m"])


def _synthetic_gt_records(
    gt_geos: dict[int, GtObjectGeometry],
    gt_eval: list,
    plane: tuple[float, float, float, float],
) -> tuple[list[dict], list[np.ndarray], dict | None, dict | None]:
    """GT self-test inputs: GT masks + GT-derived geometry posing as predictions."""
    n, u, v = plane_basis(plane)
    records: list[dict] = []
    masks: list[np.ndarray] = []
    for gi in gt_eval:
        geo = gt_geos.get(gi.instance_id)
        if geo is None:
            continue
        yaw = np.radians(geo.footprint_yaw_deg)
        long_axis = np.cos(yaw) * u + np.sin(yaw) * v
        short_axis = -np.sin(yaw) * u + np.cos(yaw) * v
        R = np.column_stack([long_axis, short_axis, n])
        records.append({
            "candidate_id": f"gt_{gi.instance_id}",
            "label": geo.class_name,
            "score": 1.0,
            "centroid_3d": geo.center_camera.tolist(),
            "top_surface_height_m": geo.h_top,
            "bottom_z_m": geo.h_bottom,
            "height_m": geo.height_m,
            "bottom_method": "measured",
            "bottom_confidence": 1.0,
            "neighbor_source": None,
            "parcel_obb": {
                "center": geo.center_camera.tolist(),
                "extents": [geo.footprint_long_m, geo.footprint_short_m, geo.height_m],
                "R": R.tolist(),
                "corners_3d": [],
            },
        })
        masks.append(gi.mask)

    if not records:
        return records, masks, None, None

    stage10 = {"primary": {"candidate": {"candidate_id": records[0]["candidate_id"]}}}
    first_geo = gt_geos[gt_eval[0].instance_id]
    stage11 = {
        "primary_grasp": {"normal": first_geo.top_normal.tolist()},
        "n_grasps": 1,
    }
    return records, masks, stage10, stage11


def evaluate_scene(
    session_path: Path,
    eval_cfg: dict,
    visibility: VisibilityFilter,
    *,
    gt_self_test: bool = False,
) -> dict:
    """Returns {"status", "candidate_rows", "grasp_row"} for one scene."""
    scene_id = session_path.name
    meta = scene_meta(session_path)
    band = meta.category

    prep = _load_json(session_path / "stage_prep_context.json")
    if prep is None:
        return {
            "status": "missing_pipeline_output",
            "candidate_rows": [],
            "grasp_row": {
                "scene_id": scene_id, "category_band": band, "gt_class": "",
                "theta_deg": "", "within_12deg": "", "within_30deg": "",
                "status": "missing_pipeline_output",
            },
        }

    plane = tuple(float(x) for x in prep["plane_model"])
    h, w = int(prep["height"]), int(prep["width"])
    ws = _decode_rle(prep["workspace_mask_rle"], h, w).astype(bool)
    fx, fy = float(prep["fx"]), float(prep["fy"])
    degenerate_plane = _is_degenerate_plane_fit(prep)

    gt_scene = build_gt_scene(
        session_path,
        ws,
        visibility=visibility,
        workspace_majority_threshold=float(
            eval_cfg.get("workspace_majority_threshold", 0.5)
        ),
    )
    gt_geos = load_gt_geometries(session_path, plane)

    stage8 = _load_json(session_path / "stage8_candidates.json")
    records = stage8["candidates"] if stage8 else []
    pred_masks = [
        _decode_rle(r["mask_rle"], h, w) for r in records
    ]
    stage10 = _load_json(session_path / "stage10_selected_target.json")
    stage11 = _load_json(session_path / "stage11_suction_grasps.json")

    if gt_self_test:
        records, pred_masks, stage10, stage11 = _synthetic_gt_records(
            gt_geos, gt_scene.gt_eval, plane
        )

    pred_clipped = [clip_mask(m, ws) for m in pred_masks]
    scores = np.asarray([float(r.get("score", 0.0)) for r in records], dtype=np.float64)
    gt_masks = [g.mask for g in gt_scene.gt_eval]

    ious = iou_matrix(pred_clipped, gt_masks)
    match = greedy_match(ious, scores, float(eval_cfg.get("iou_threshold", 0.5)))

    soft_classes = set(eval_cfg.get("soft_classes", []))
    primary_id = None
    if stage10 and stage10.get("primary"):
        primary_id = stage10["primary"]["candidate"]["candidate_id"]

    matched_gt_by_pred: dict[int, tuple[int, float]] = {
        p: (g, iou) for p, g, iou in match.tp_pairs
    }

    candidate_rows: list[dict] = []
    for p, (g, iou) in matched_gt_by_pred.items():
        gi = gt_scene.gt_eval[g]
        geo = gt_geos.get(gi.instance_id)
        if geo is None:
            continue
        pred_geo = derive_pred_geometry(records[p], plane)
        err = candidate_errors(pred_geo, geo)
        cue = err.bottom_cue
        # In a broken plane frame, the from_pallet snap (bottom_z := 0) is a
        # pipeline error of camera-height magnitude; keep it out of the
        # regular fallback_pallet cue row.
        if degenerate_plane and cue == "fallback_pallet":
            cue = "fallback_pallet_degenerate_plane"
        candidate_rows.append({
            # candidate_id is internal (spot-check join); not in the CSV schema.
            "candidate_id": records[p]["candidate_id"],
            "scene_id": scene_id,
            "category_band": band,
            "gt_instance_id": gi.instance_id,
            "gt_class": geo.class_name,
            "packaging_type": "soft" if geo.class_name in soft_classes else "rigid",
            "visible_px": gi.visible_pixels,
            "visibility_ratio": visibility_ratio(geo, fx, fy),
            "match_iou": float(iou),
            "e_lat_mm": err.e_lat_mm,
            "e_top_mm_signed": err.e_top_mm_signed,
            "ext_err_long_rel": err.ext_err_long_rel,
            "ext_err_short_rel": err.ext_err_short_rel,
            "ext_err_height_rel": err.ext_err_height_rel,
            "yaw_err_deg": err.yaw_err_deg,
            "yaw_fold_deg": err.yaw_fold_deg,
            "e_bottom_mm_signed": err.e_bottom_mm_signed,
            "bottom_cue": cue,
            "bottom_confidence": err.bottom_confidence,
            "is_primary_target": records[p]["candidate_id"] == primary_id,
        })

    grasp_row = _grasp_row(
        scene_id, band, records, matched_gt_by_pred, gt_scene, gt_geos,
        primary_id, stage11, eval_cfg,
    )

    return {
        "status": "ok",
        "candidate_rows": candidate_rows,
        "grasp_row": grasp_row,
        "degenerate_plane_fit": degenerate_plane,
    }


def _grasp_row(
    scene_id: str,
    band: str,
    records: list[dict],
    matched_gt_by_pred: dict[int, tuple[int, float]],
    gt_scene,
    gt_geos: dict[int, GtObjectGeometry],
    primary_id: str | None,
    stage11: dict | None,
    eval_cfg: dict,
) -> dict:
    """Scene-level theta row for the primary grasp (spec 6.3 / 8.2)."""
    row = {
        "scene_id": scene_id, "category_band": band, "gt_class": "",
        "theta_deg": "", "within_12deg": "", "within_30deg": "", "status": "",
    }
    if primary_id is None:
        row["status"] = "no_target"
        return row

    primary_pred_idx = next(
        (i for i, r in enumerate(records) if r["candidate_id"] == primary_id), None
    )
    if primary_pred_idx is None:
        # stage10 primary not present in stage8 of the same run -> stale file.
        row["status"] = "stale_target_record"
        return row
    if primary_pred_idx not in matched_gt_by_pred:
        row["status"] = "target_unmatched"
        return row

    primary_grasp = (stage11 or {}).get("primary_grasp")
    if not primary_grasp or not primary_grasp.get("normal"):
        row["status"] = "no_grasp"
        return row

    g, _ = matched_gt_by_pred[primary_pred_idx]
    gi = gt_scene.gt_eval[g]
    geo = gt_geos[gi.instance_id]

    n_pred = orient_toward_camera(np.asarray(primary_grasp["normal"], dtype=np.float64))
    n_gt = orient_toward_camera(geo.top_normal)
    theta = angle_between_deg(n_pred, n_gt)

    t12, t30 = eval_cfg.get("rate_theta_deg", [12.0, 30.0])
    row.update({
        "gt_class": geo.class_name,
        "theta_deg": theta,
        "within_12deg": theta <= float(t12),
        "within_30deg": theta <= float(t30),
        "status": "evaluated",
    })
    return row


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


def _maybe_write_figure(grasp_rows: list[dict], band_order: list[str], fig_root: Path) -> None:
    """Per-band theta distribution (box plot), only if the sample is readable."""
    evaluated = [r for r in grasp_rows if r["status"] == "evaluated"]
    by_band = {
        b: [r["theta_deg"] for r in evaluated if r["category_band"] == b]
        for b in band_order
    }
    populated = {b: v for b, v in by_band.items() if len(v) >= 3}
    if len(evaluated) < 30 or len(populated) < 3:
        print(
            f"[FIGURE] übersprungen: theta-Stichprobe zu klein "
            f"(n={len(evaluated)}, Bänder mit >=3: {len(populated)})"
        )
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(populated.keys())
    ax.boxplot([populated[b] for b in labels], tick_labels=labels, showfliers=True)
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_xlabel("category band")
    ax.axhline(12.0, color="tab:orange", linestyle="--", linewidth=0.8, label="12°")
    ax.axhline(30.0, color="tab:red", linestyle="--", linewidth=0.8, label="30°")
    ax.legend(loc="upper right", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig_dir = fig_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "exp2_normal_deviation.pdf"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[FIGURE] geschrieben -> {fig_path}")


def _assert_gt_self_test(candidate_rows: list[dict], grasp_rows: list[dict]) -> None:
    """Every error must vanish, every rate must be 1 (spec 9.1)."""
    tol = 1e-6
    failures = []
    for r in candidate_rows:
        checks = {
            "e_lat_mm": r["e_lat_mm"],
            "e_top_mm_signed": r["e_top_mm_signed"],
            "ext_err_long_rel": r["ext_err_long_rel"],
            "ext_err_short_rel": r["ext_err_short_rel"],
            "ext_err_height_rel": r["ext_err_height_rel"],
            "yaw_err_deg": r["yaw_err_deg"],
            "e_bottom_mm_signed": r["e_bottom_mm_signed"],
        }
        for name, val in checks.items():
            if val is None or abs(float(val)) > tol:
                failures.append(f"{r['scene_id']}/gt_{r['gt_instance_id']}: {name}={val}")
        if abs(r["match_iou"] - 1.0) > tol:
            failures.append(f"{r['scene_id']}/gt_{r['gt_instance_id']}: iou={r['match_iou']}")
    for r in grasp_rows:
        if r["status"] == "evaluated" and abs(float(r["theta_deg"])) > tol:
            failures.append(f"{r['scene_id']}: theta={r['theta_deg']}")
    if failures:
        for f in failures[:20]:
            print(f"[GT-SELF-TEST] FAIL {f}")
        raise SystemExit(f"GT self-test failed: {len(failures)} violations")
    print(
        f"[GT-SELF-TEST] PASS — {len(candidate_rows)} Kandidaten, "
        f"{sum(1 for r in grasp_rows if r['status'] == 'evaluated')} Grasps, alle Fehler < 1e-6"
    )


def run_evaluation(
    session_paths: list[Path],
    eval_cfg: dict,
    visibility: VisibilityFilter,
    out_dir: Path,
    *,
    label: str,
    run_id: str | None = None,
    gt_self_test: bool = False,
    fig_root: Path | None = None,
) -> dict:
    candidate_rows: list[dict] = []
    grasp_rows: list[dict] = []
    n_missing = 0
    n_degenerate = 0

    for i, sp in enumerate(session_paths, start=1):
        result = evaluate_scene(sp, eval_cfg, visibility, gt_self_test=gt_self_test)
        if result["status"] == "missing_pipeline_output":
            n_missing += 1
        if result.get("degenerate_plane_fit"):
            n_degenerate += 1
        candidate_rows.extend(result["candidate_rows"])
        grasp_rows.append(result["grasp_row"])
        if i % 100 == 0 or i == len(session_paths):
            print(f"[{label}] {i}/{len(session_paths)} Szenen — "
                  f"{len(candidate_rows)} gematchte Kandidaten")

    if gt_self_test:
        _assert_gt_self_test(candidate_rows, grasp_rows)

    band_order = list(eval_cfg.get("category_rows", []))
    meta = {
        "run_id": run_id or out_dir.name,
        "config_hash": config_hash(eval_cfg),
        "git_commit": _git_commit(),
        "visibility": {
            "mode": visibility.mode,
            "absolute_min": visibility.absolute_min,
            "relative_fraction": visibility.relative_fraction,
        },
        "n_scenes": len(session_paths),
        "n_scenes_missing_pipeline_output": n_missing,
        "n_scenes_degenerate_plane_fit": n_degenerate,
    }
    summary = build_summary(candidate_rows, grasp_rows, band_order, meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "exp2_per_candidate.csv", candidate_rows, PER_CANDIDATE_COLUMNS)
    _write_csv(out_dir / "exp2_per_grasp.csv", grasp_rows, PER_GRASP_COLUMNS)
    with open(out_dir / "exp2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[{label}] Summary -> {out_dir / 'exp2_summary.json'}")

    if fig_root is not None and not gt_self_test:
        _maybe_write_figure(grasp_rows, band_order, fig_root)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 — offline evaluation")
    parser.add_argument(
        "--data-root", type=Path,
        default=PROJECT_ROOT / "Data" / "blender_dataset",
        help="Dataset root with scene_XXX directories",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: Results/exp2/<timestamp>)",
    )
    parser.add_argument(
        "--visibility", choices=["primary", "strict", "both"], default="both",
        help="Visibility rule(s), mirroring Experiment 1 (default: both)",
    )
    parser.add_argument("--limit", type=int, default=None, help="First N scenes only")
    parser.add_argument(
        "--scenes", type=str, default=None,
        help="Comma-separated scene ids (e.g. scene_000,scene_042)",
    )
    parser.add_argument(
        "--gt-self-test", action="store_true",
        help="Feed GT-derived quantities as predictions; assert zero errors",
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
    print(f"[EXP2] {len(sessions)} Szenen aus {args.data_root}")

    if args.out_dir is None:
        from datetime import datetime

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.out_dir = PROJECT_ROOT / "Results" / "exp2" / stamp

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
            gt_self_test=args.gt_self_test,
            # One figure per run, from the first (primary) evaluation.
            fig_root=args.out_dir if i == 0 else None,
        )


if __name__ == "__main__":
    main()
