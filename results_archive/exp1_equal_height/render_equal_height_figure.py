"""Renders thesis Figure 10.4 (equal-height example, scene 408, pair 3/9)
from the archived predictions and the SynDePal ground truth.

Run from the repository root:
    python results_archive/exp1_equal_height/render_equal_height_figure.py
Writes exp1_equal_height_example.pdf next to this script.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.masks import decode_masks_rle  # noqa: E402

sid = "scene_408"
PAIR = (3, 9)
sp = ROOT / "Data/blender_dataset" / sid
rgb = plt.imread(sp / "rgb.png")
inst = np.load(sp / "instance_mask.npy")
ARCHIVE = ROOT / "results_archive"
runs = [("(b) Standard pipeline: masks straddle the seam",
         ARCHIVE / "exp1_segmentation_standard/preds"),
        ("(c) SAM variant: one mask per parcel",
         ARCHIVE / "exp1_segmentation_sam_variant/preds")]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
ax = axes[0]
ax.imshow(rgb)
for k, oid in enumerate(PAIR):
    ax.contour(inst == oid, levels=[0.5],
               colors=[["#2166ac", "#b2182b"][k]], linewidths=2)
ax.set_title("(a) Ground truth: two flush parcels", fontsize=10)
cmap = plt.get_cmap("tab10")
for ax, (title, pdir) in zip(axes[1:], runs):
    z = np.load(pdir / f"{sid}.npz", allow_pickle=True)
    preds = decode_masks_rle(z["masks_F_rle"], int(z["height"]), int(z["width"]))
    gt_a, gt_b = (inst == PAIR[0]), (inst == PAIR[1])
    ax.imshow(rgb)
    shown = 0
    for p in preds:
        pb = p.astype(bool)
        if ((gt_a & pb).sum() / max(gt_a.sum(), 1) > 0.1
                or (gt_b & pb).sum() / max(gt_b.sum(), 1) > 0.1
                or (pb & (gt_a | gt_b)).sum() / max(pb.sum(), 1) > 0.05):
            ax.imshow(np.dstack([np.full(pb.shape, c) for c in cmap(shown % 10)[:3]]
                                + [pb * 0.45]), interpolation="nearest")
            ax.contour(pb, levels=[0.5], colors=[cmap(shown % 10)], linewidths=1.5)
            shown += 1
    for oid in PAIR:
        ax.contour(inst == oid, levels=[0.5], colors=["white"],
                   linewidths=0.8, linestyles="dashed")
    ax.set_title(title, fontsize=10)
ys, xs = np.where((inst == PAIR[0]) | (inst == PAIR[1]))
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(max(xs.min() - 90, 0), min(xs.max() + 90, rgb.shape[1]))
    ax.set_ylim(min(ys.max() + 90, rgb.shape[0]), max(ys.min() - 90, 0))
plt.tight_layout()
out = Path(__file__).parent / "exp1_equal_height_example.pdf"
plt.savefig(out, bbox_inches="tight", dpi=200)
print("saved", out)
