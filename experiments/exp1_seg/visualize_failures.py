"""Qualitative failure-mode overlays for Experiment 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_PATH  # noqa: E402
from evaluation.gt import build_gt_scene, load_instance_mask  # noqa: E402
from evaluation.masks import decode_masks_rle, iou_matrix  # noqa: E402
from evaluation.matching import greedy_match  # noqa: E402
from experiments.exp1_seg.evaluate import _load_preds  # noqa: E402


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45):
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def classify_failure(pred_masks, scores, gt_masks) -> str:
    ious = iou_matrix(pred_masks, gt_masks)
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_pred == 0 and n_gt > 0:
        return "miss"
    if n_gt == 0 and n_pred > 0:
        return "hallucination"
    if n_gt and np.any(np.sum(ious >= 0.10, axis=0) >= 2):
        return "over_segmentation"
    if n_pred and np.any(np.sum(ious >= 0.10, axis=1) >= 2):
        return "merge"
    if n_pred and np.all(np.max(ious, axis=1) < 0.10):
        return "hallucination"
    if n_gt and np.any(np.max(ious, axis=0) < 0.50):
        return "miss"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export failure-mode qualitative overlays")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {"over_segmentation", "merge", "hallucination", "miss"}
    found: dict[str, str] = {}

    for npz_path in sorted((run_dir / "preds").glob("scene_*.npz")):
        scene_id = npz_path.stem
        preds = _load_preds(npz_path)
        session = Path(BASE_PATH) / scene_id
        ws = preds["workspace_mask"]
        gt_scene = build_gt_scene(session, ws)
        gt_masks = [g.mask for g in gt_scene.gt_eval]
        mode = classify_failure(preds["masks_F"], preds["scores_F"], gt_masks)
        if mode in targets and mode not in found:
            found[mode] = scene_id
        if len(found) == len(targets):
            break

    for mode, scene_id in found.items():
        session = Path(BASE_PATH) / scene_id
        npz_path = run_dir / "preds" / f"{scene_id}.npz"
        preds = _load_preds(npz_path)
        rgb = cv2.cvtColor(cv2.imread(str(session / "rgb.png")), cv2.COLOR_BGR2RGB)
        inst = load_instance_mask(session)
        gt_vis = np.zeros_like(rgb)
        for lid in np.unique(inst):
            if lid < 0:
                continue
            gt_vis[inst == lid] = (0, 200, 0)
        canvas = _overlay(rgb, gt_vis.any(axis=2), (0, 200, 0))
        for m in preds["masks_F"]:
            canvas = _overlay(canvas, m, (255, 80, 0))
        out_path = out_dir / f"{mode}_{scene_id}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"[FIG] {mode} -> {out_path}")


if __name__ == "__main__":
    main()
