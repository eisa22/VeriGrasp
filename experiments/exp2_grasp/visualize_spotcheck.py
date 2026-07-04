"""Experiment 2 spot check: overlay predicted centroid, GT centre, and normals.

Dumps RGB overlays for N random matched candidates from different bands
(acceptance criterion 9.3). Predicted centroid is drawn as a filled circle,
the GT box centre as a cross; the primary-grasp approach normal and the GT
top-face normal are drawn as projected arrows for primary targets.

Usage:
    python -m experiments.exp2_grasp.visualize_spotcheck \
        [--n 5] [--seed 0] [--data-root ...] [--out-dir figures/exp2_spotcheck]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.exp2_geometry import orient_toward_camera  # noqa: E402
from evaluation.exp2_gt import load_gt_geometries  # noqa: E402
from evaluation.gt import VisibilityFilter  # noqa: E402
from experiments.exp2_grasp.evaluate import (  # noqa: E402
    _load_eval_config,
    _load_json,
    evaluate_scene,
)

COLOR_PRED = (60, 200, 60)    # BGR green: prediction
COLOR_GT = (60, 60, 230)      # BGR red: ground truth


def _project(p3d: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[int, int]:
    x, y, z = float(p3d[0]), float(p3d[1]), float(p3d[2])
    return int(round(fx * x / z + cx)), int(round(fy * y / z + cy))


def _draw_normal(img, origin_3d, normal, fx, fy, cx, cy, color, length_m=0.15):
    tip = np.asarray(origin_3d) + orient_toward_camera(normal) * length_m
    p0 = _project(np.asarray(origin_3d), fx, fy, cx, cy)
    p1 = _project(tip, fx, fy, cx, cy)
    cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.25)


def render_scene(session_path: Path, row: dict, out_path: Path) -> None:
    rgb = cv2.imread(str(session_path / "rgb.png"))
    prep = _load_json(session_path / "stage_prep_context.json")
    fx, fy, cx, cy = prep["fx"], prep["fy"], prep["cx"], prep["cy"]
    plane = tuple(float(x) for x in prep["plane_model"])

    stage8 = _load_json(session_path / "stage8_candidates.json") or {"candidates": []}
    gt_geos = load_gt_geometries(session_path, plane)
    geo = gt_geos[row["gt_instance_id"]]

    # GT centre (cross, red) + GT top-face normal
    gx, gy = _project(geo.center_camera, fx, fy, cx, cy)
    cv2.drawMarker(rgb, (gx, gy), COLOR_GT, cv2.MARKER_CROSS, 18, 2)
    _draw_normal(rgb, geo.center_camera, geo.top_normal, fx, fy, cx, cy, COLOR_GT)

    # Predicted centroid (circle, green), located via the matched candidate id.
    record = next(
        (r for r in stage8["candidates"] if r["candidate_id"] == row["candidate_id"]),
        None,
    )
    if record is not None:
        c = np.asarray(record["centroid_3d"], dtype=np.float64)
        px, py = _project(c, fx, fy, cx, cy)
        cv2.circle(rgb, (px, py), 7, COLOR_PRED, 2)
        if row.get("is_primary_target"):
            stage11 = _load_json(session_path / "stage11_suction_grasps.json")
            pg = (stage11 or {}).get("primary_grasp")
            if pg and pg.get("normal"):
                _draw_normal(
                    rgb, np.asarray(pg["position"]), np.asarray(pg["normal"]),
                    fx, fy, cx, cy, COLOR_PRED,
                )

    label = (
        f"{row['scene_id']} [{row['category_band']}] {row['gt_class']} "
        f"e_lat={row['e_lat_mm']:.1f}mm e_top={row['e_top_mm_signed']:+.1f}mm"
    )
    cv2.putText(rgb, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), rgb)
    print(f"[SPOTCHECK] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 spot-check overlays")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-root", type=Path,
        default=PROJECT_ROOT / "Data" / "blender_dataset",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "Results" / "exp2" / "figures" / "exp2_spotcheck",
    )
    args = parser.parse_args()

    eval_cfg = _load_eval_config()
    vis = VisibilityFilter(mode="absolute", absolute_min=1)
    rng = random.Random(args.seed)

    sessions = sorted(
        p for p in args.data_root.iterdir()
        if p.is_dir() and p.name.startswith("scene_")
        and (p / "stage8_candidates.json").exists()
    )
    rng.shuffle(sessions)

    # Collect matched candidates until we have n from distinct bands.
    rows_by_band: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for sp in sessions:
        result = evaluate_scene(sp, eval_cfg, vis)
        for row in result["candidate_rows"]:
            rows_by_band[row["category_band"]].append((sp, row))
        if sum(1 for v in rows_by_band.values() if v) >= args.n:
            break

    picks: list[tuple[Path, dict]] = []
    for band in sorted(rows_by_band.keys()):
        picks.append(rng.choice(rows_by_band[band]))
        if len(picks) >= args.n:
            break

    if not picks:
        raise SystemExit("Keine gematchten Kandidaten gefunden — Pipeline-Lauf fehlt?")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sp, row in picks:
        out = args.out_dir / f"{row['scene_id']}_gt{row['gt_instance_id']}.png"
        render_scene(sp, row, out)


if __name__ == "__main__":
    main()
