"""Renders the qualitative result figures of the thesis (success/failure
gallery and standard-vs-SAM comparison) from the archived predictions.

Run from the repository root:
    python results_archive/exp1_equal_height/render_qualitative_figures.py
Writes exp1_qualitative_failures.pdf and exp1_std_vs_sam_dense.pdf next to
this script. Requires the SynDePal ground truth under Data/blender_dataset.
Scene selection: scene_020 (max TP at 0 FP in the baseline band),
scene_250/100/578 (failure cases), scene_327 (largest SAM-vs-standard
TP gap in the dense band), all read from the archived per-scene metrics.
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

ARCH = ROOT / "results_archive"
DATA = ROOT / "Data/blender_dataset"
OUT = Path(__file__).parent
CMAP = plt.get_cmap("tab10")

def load(run, sid):
    z = np.load(ARCH / run / "preds" / f"{sid}.npz", allow_pickle=True)
    H, W = int(z["height"]), int(z["width"])
    preds = [p.astype(bool) for p in decode_masks_rle(z["masks_F_rle"], H, W)]
    ws = z["workspace_mask"].astype(bool)
    return [p & ws for p in preds if (p & ws).any()], ws

def composite(rgb, preds):
    img = rgb[..., :3].astype(float)
    if img.max() > 1.5:
        img /= 255.0
    for i, m in enumerate(preds):
        c = np.array(CMAP(i % 10)[:3])
        img[m] = 0.55 * img[m] + 0.45 * c
    return img

def panel(ax, sid, run, title):
    rgb = plt.imread(DATA / sid / "rgb.png")
    inst = np.load(DATA / sid / "instance_mask.npy")
    preds, ws = load(run, sid)
    ax.imshow(composite(rgb, preds))
    for m in preds:
        ax.contour(m, levels=[0.5], colors=["black"], linewidths=0.6)
    gt_ids = [i for i in np.unique(inst) if i >= 0 and ((inst == i) & ws).sum() >= 1]
    for g in gt_ids:
        ax.contour((inst == g) & ws, levels=[0.5], colors=["white"],
                   linewidths=0.7, linestyles="dashed")
    ax.set_title(f"{title}\n{len(preds)} mask{'s' if len(preds)!=1 else ''} / "
                 f"{len(gt_ids)} visible parcels", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ys, xs = np.where(ws)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.max(), ys.min())

fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
panel(axes[0, 0], "scene_020", "exp1_segmentation_standard", "(a) baseline: success case")
panel(axes[0, 1], "scene_250", "exp1_segmentation_standard", "(b) dense stack: under-detection")
panel(axes[1, 0], "scene_100", "exp1_segmentation_standard", "(c) mixed: non-box packaging, no output")
panel(axes[1, 1], "scene_578", "exp1_segmentation_standard", "(d) angled view: no detection survives")
plt.tight_layout()
plt.savefig(OUT / "exp1_qualitative_failures.pdf", bbox_inches="tight", dpi=200)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
panel(axes[0], "scene_327", "exp1_segmentation_standard", "(a) standard pipeline")
panel(axes[1], "scene_327", "exp1_segmentation_sam_variant", "(b) SAM variant")
plt.tight_layout()
plt.savefig(OUT / "exp1_std_vs_sam_dense.pdf", bbox_inches="tight", dpi=200)
print("saved", OUT)
