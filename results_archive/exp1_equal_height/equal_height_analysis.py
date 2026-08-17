"""Equal-height adjacent-pair analysis over the archived Exp1 predictions.

Produces the counts of thesis Table 10.5 (equal-height cases at the output
stage) from the archived per-scene predictions of both Experiment-1 runs.
Requires the SynDePal ground truth under Data/blender_dataset/ (distributed
with the thesis, not part of this repository).

Run from the repository root:
    python results_archive/exp1_equal_height/equal_height_analysis.py

Definitions (as stated in the thesis table caption):
- Pair: two evaluable GT parcels (>=50 visible px each, workspace-clipped),
  top-height difference <= 10 mm (exact GT bbox corners), and 2D-adjacent
  (5 px dilation of one visible mask intersects the other).
- Separated: both members matched by (distinct) predictions at IoU >= 0.5
  under greedy score-ordered unique matching (same rule as the evaluation).
- Merged: at least one prediction has IoU >= 0.10 with BOTH members
  (same 0.10 partial-contact rule as the failure-mode metrics).
- False split (singles subset): an evaluable parcel (>=50 px) that is not a
  member of any equal-height pair and is covered by >= 2 predictions at
  IoU >= 0.10 (the over-segmentation counting rule).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.gt import build_gt_scene  # noqa: E402
from evaluation.masks import decode_masks_rle  # noqa: E402

DATA = ROOT / "Data/blender_dataset"
ARCHIVE = ROOT / "results_archive"
RUNS = {
    "std": ARCHIVE / "exp1_segmentation_standard/preds",
    "sam": ARCHIVE / "exp1_segmentation_sam_variant/preds",
}
TOL_M = 0.010
MIN_PIX = 50
KERNEL = np.ones((11, 11), np.uint8)  # 5 px dilation radius

def iou_matrix(preds, gts):
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    P = np.stack([p.astype(bool).ravel() for p in preds])
    G = np.stack([g.astype(bool).ravel() for g in gts])
    inter = (P.astype(np.uint64) @ G.T.astype(np.uint64)).astype(float)
    pa = P.sum(1, dtype=np.uint64).astype(float)[:, None]
    ga = G.sum(1, dtype=np.uint64).astype(float)[None, :]
    union = pa + ga - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out

def greedy_matched_gt(ious, scores, thr=0.5):
    matched = set()
    used_pred = set()
    order = np.argsort(-np.asarray(scores)) if len(scores) else []
    for p in order:
        if p in used_pred:
            continue
        free = [g for g in range(ious.shape[1]) if g not in matched]
        if not free:
            break
        best_g, best_iou = -1, thr
        for g in free:
            if ious[p, g] >= best_iou:
                best_g, best_iou = g, ious[p, g]
        if best_g >= 0:
            matched.add(best_g)
            used_pred.add(p)
    return matched

totals = {
    "pairs": 0, "singles": 0,
    "std": {"sep": 0, "merged": 0, "neither": 0, "false_split": 0},
    "sam": {"sep": 0, "merged": 0, "neither": 0, "false_split": 0},
}
per_band_pairs = {}
examples = []  # scenes where sam separates and std merges

scenes = sorted(p for p in DATA.iterdir() if p.name.startswith("scene_"))
for si, sp in enumerate(scenes):
    sid = sp.name
    npz = {k: np.load(RUNS[k] / f"{sid}.npz", allow_pickle=True) for k in RUNS}
    ws = npz["std"]["workspace_mask"].astype(bool)
    H, W = ws.shape
    scene = build_gt_scene(sp, ws, assert_reconciliation=False)
    gt = json.load(open(sp / "ground_truth.json"))
    top_z = {}
    for o in gt["objects"]:
        corners = np.array(o["bbox_corners_camera_frame"])
        top_z[o["id"]] = float(corners[:, 2].min())
    insts = [i for i in scene.gt_eval if i.visible_pixels >= MIN_PIX and i.instance_id in top_z]
    masks = {i.instance_id: (i.mask.astype(bool) & ws) for i in insts}
    dil = {k: cv2.dilate(m.astype(np.uint8), KERNEL).astype(bool) for k, m in masks.items()}
    ids = [i.instance_id for i in insts]
    pairs = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if abs(top_z[i] - top_z[j]) <= TOL_M and (dil[i] & masks[j]).any():
                pairs.append((i, j))
    in_pair = {i for p in pairs for i in p}
    singles = [i for i in ids if i not in in_pair]
    totals["pairs"] += len(pairs)
    totals["singles"] += len(singles)
    n_scene = int(sid.split("_")[1])
    band = ("baseline" if n_scene < 56 else "mixed" if n_scene < 200 else
            "dense" if n_scene < 350 else "chaotic" if n_scene < 430 else
            "lighting" if n_scene < 480 else "occlusion" if n_scene < 530 else
            "edge" if n_scene < 560 else "tilted" if n_scene < 578 else "angled")
    per_band_pairs[band] = per_band_pairs.get(band, 0) + len(pairs)

    outcome = {}
    for run in RUNS:
        z = npz[run]
        preds = [(p.astype(bool) & ws) for p in decode_masks_rle(z["masks_F_rle"], H, W)]
        preds = [p for p in preds if p.any()]
        scores = list(z["scores_F"])[: len(preds)]
        gt_masks = [masks[i] for i in ids]
        ious = iou_matrix(preds, gt_masks)
        matched_idx = greedy_matched_gt(ious, scores) if len(preds) else set()
        matched_ids = {ids[g] for g in matched_idx}
        res = {"sep": [], "merged": []}
        for (i, j) in pairs:
            gi, gj = ids.index(i), ids.index(j)
            sep = i in matched_ids and j in matched_ids
            mrg = bool(len(preds)) and bool(
                np.any((ious[:, gi] >= 0.10) & (ious[:, gj] >= 0.10)))
            if sep:
                totals[run]["sep"] += 1
                res["sep"].append((i, j))
            elif mrg:
                totals[run]["merged"] += 1
                res["merged"].append((i, j))
            else:
                totals[run]["neither"] += 1
        for i in singles:
            gi = ids.index(i)
            if len(preds) and int(np.sum(ious[:, gi] >= 0.10)) >= 2:
                totals[run]["false_split"] += 1
        outcome[run] = res
    for (i, j) in pairs:
        if (i, j) in outcome["sam"]["sep"] and (i, j) in outcome["std"]["merged"]:
            examples.append({"scene": sid, "pair": [i, j], "band": band})
    if (si + 1) % 100 == 0:
        print(f"{si+1}/728 pairs={totals['pairs']}", flush=True)

out = {"totals": totals, "per_band_pairs": per_band_pairs,
       "examples_sam_sep_std_merged": examples[:40],
       "n_examples": len(examples),
       "params": {"tol_mm": 10, "min_pix": MIN_PIX, "dilate_px": 5}}
outp = Path(__file__).parent / "equal_height_result.json"
json.dump(out, open(outp, "w"), indent=1)
print(json.dumps(totals, indent=1))
print("bands:", per_band_pairs)
print("saved", outp)
