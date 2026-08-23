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

# ---------------------------------------------------------------------------
# Additional figures: stage progression (scene_020) and invisible seam
# (scene_408, pair 3/9). Same data sources as above.
# ---------------------------------------------------------------------------
import cv2
import matplotlib.patches as mpatches

sid = "scene_020"
z = np.load(ARCH / "exp1_segmentation_standard/preds" / f"{sid}.npz", allow_pickle=True)
H, W = int(z["height"]), int(z["width"])
ws = z["workspace_mask"].astype(bool)
rgb = plt.imread(DATA / sid / "rgb.png")
stages = {k: [m.astype(bool) & ws for m in decode_masks_rle(z[f"masks_{k}_rle"], H, W)]
          for k in ("S", "M", "F")}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
axes[0].imshow(rgb[..., :3])
for i, b in enumerate(z["boxes_D"]):
    x0, y0, x1, y1 = b
    axes[0].add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                         edgecolor=CMAP(i % 10), linewidth=1.8))
axes[0].set_title(f"(a) D: detection\n{len(z['boxes_D'])} boxes", fontsize=10)
for ax, (name, label) in zip(axes[1:], [("S", "gradient segments"),
                                        ("M", "matched masks"), ("F", "refined masks")]):
    ms = [m for m in stages[name] if m.any()]
    ax.imshow(composite(rgb, ms))
    for m in ms:
        ax.contour(m, levels=[0.5], colors=["black"], linewidths=0.6)
    lett = {"S": "b", "M": "c", "F": "d"}[name]
    ax.set_title(f"({lett}) {name}: {label}\n{len(ms)} masks", fontsize=10)
ys, xs = np.where(ws)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.max(), ys.min())
plt.tight_layout()
plt.savefig(OUT / "exp1_stage_progression.pdf", bbox_inches="tight", dpi=200)

sid, PAIR = "scene_408", (3, 9)
rgb = plt.imread(DATA / sid / "rgb.png")
inst = np.load(DATA / sid / "instance_mask.npy")
depth = np.load(DATA / sid / "depth.npy").astype(float)
gx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
grad = np.sqrt(gx ** 2 + gy ** 2)
ys, xs = np.where((inst == PAIR[0]) | (inst == PAIR[1]))
y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 40, depth.shape[0])
x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 40, depth.shape[1])
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].imshow(rgb[y0:y1, x0:x1, :3])
for k, oid in enumerate(PAIR):
    axes[0].contour((inst == oid)[y0:y1, x0:x1], levels=[0.5],
                    colors=[["#2166ac", "#b2182b"][k]], linewidths=2)
axes[0].set_title("(a) RGB: two parcels, visible seam", fontsize=10)
im1 = axes[1].imshow(depth[y0:y1, x0:x1], cmap="viridis")
axes[1].set_title("(b) depth: no step at the seam", fontsize=10)
plt.colorbar(im1, ax=axes[1], fraction=0.046, label="depth [m]")
gcrop = grad[y0:y1, x0:x1]
im2 = axes[2].imshow(np.clip(gcrop, 0, np.percentile(gcrop, 99)), cmap="magma")
axes[2].set_title("(c) depth gradient: seam invisible", fontsize=10)
plt.colorbar(im2, ax=axes[2], fraction=0.046, label="|grad z|")
for k, oid in enumerate(PAIR):
    for ax in axes[1:]:
        ax.contour((inst == oid)[y0:y1, x0:x1], levels=[0.5], colors=["white"],
                   linewidths=0.8, linestyles="dashed")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(OUT / "exp1_invisible_seam.pdf", bbox_inches="tight", dpi=200)
print("saved additional figures", OUT)

# ---------------------------------------------------------------------------
# Additional figures: intro scene example (scene_020), SAM-specific failures
# (scene_200 bleed / scene_089 hallucination), and one grasp per verification
# outcome (scenes 007/204/073/009, from exp3_per_grasp.csv). Grasp pixels are
# read from the persisted stage11_suction_grasps.json of each scene; scene
# selection for the verdict figure requires the grasp pixel to lie inside a
# stage-F mask of the archived standard run.
# ---------------------------------------------------------------------------
import json

def _std(sid):
    z = np.load(ARCH / "exp1_segmentation_standard/preds" / f"{sid}.npz", allow_pickle=True)
    H, W = int(z["height"]), int(z["width"])
    preds = [p.astype(bool) for p in decode_masks_rle(z["masks_F_rle"], H, W)]
    ws = z["workspace_mask"].astype(bool)
    return [p & ws for p in preds if (p & ws).any()], ws

def _grasp_px(sid):
    return json.load(open(DATA / sid / "stage11_suction_grasps.json"))["primary_grasp"]["pixel"]

# Intro: input vs output (scene_020; ACCEPT and oracle-valid per exp3 CSV)
sid = "scene_020"
rgb = plt.imread(DATA / sid / "rgb.png")
preds, ws = _std(sid)
u, v = _grasp_px(sid)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].imshow(rgb[..., :3])
axes[0].set_title("(a) input: RGB-D view of an unknown parcel stack", fontsize=10)
axes[1].imshow(composite(rgb, preds))
for m in preds:
    axes[1].contour(m, levels=[0.5], colors=["black"], linewidths=0.6)
axes[1].scatter([u], [v], s=380, facecolors="none", edgecolors="#1a9641", linewidths=3)
axes[1].scatter([u], [v], s=28, c="#1a9641", marker="x", linewidths=2.5)
axes[1].set_title("(b) output: parcel masks and the verified grasp", fontsize=10)
ys, xs = np.where(ws)
for ax in axes:
    ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.max(), ys.min())
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(OUT / "intro_scene_example.pdf", bbox_inches="tight", dpi=200)

# SAM-specific failures (variant run): bleed and hallucination
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (sid, title, pick) in zip(axes, [
        ("scene_200", "(a) one SAM mask over two parcels", "bleed"),
        ("scene_089", "(b) hallucinated SAM mask", "hal")]):
    rgb = plt.imread(DATA / sid / "rgb.png")
    inst = np.load(DATA / sid / "instance_mask.npy")
    z = np.load(ARCH / "exp1_segmentation_sam_variant/preds" / f"{sid}.npz", allow_pickle=True)
    H, W = int(z["height"]), int(z["width"])
    ws = z["workspace_mask"].astype(bool)
    preds = [p.astype(bool) & ws for p in decode_masks_rle(z["masks_F_rle"], H, W)]
    gts = [(inst == g) & ws for g in np.unique(inst) if g >= 0 and ((inst == g) & ws).sum() >= 300]
    def iou(a, b):
        u_ = (a | b).sum(); return (a & b).sum() / u_ if u_ else 0.0
    pm = None
    for p in preds:
        io = sorted((iou(p, gm) for gm in gts), reverse=True)
        if pick == "bleed" and len(io) >= 2 and io[0] >= 0.25 and io[1] >= 0.15:
            pm = p; break
        if pick == "hal" and p.sum() >= 2000 and (not io or io[0] < 0.05):
            pm = p; break
    ax.imshow(composite(rgb, [pm]))
    ax.contour(pm, levels=[0.5], colors=["#b2182b"], linewidths=2)
    for g in np.unique(inst):
        if g >= 0 and ((inst == g) & ws).sum() >= 100:
            ax.contour((inst == g) & ws, levels=[0.5], colors=["white"],
                       linewidths=0.7, linestyles="dashed")
    ys, xs = np.where(pm)
    ax.set_xlim(max(xs.min() - 70, 0), min(xs.max() + 70, rgb.shape[1]))
    ax.set_ylim(min(ys.max() + 70, rgb.shape[0]), max(ys.min() - 70, 0))
    ax.set_title(f"{title} ({sid.replace('_', ' ')})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(OUT / "exp1_sam_failures.pdf", bbox_inches="tight", dpi=200)

# One grasp per verification outcome (Experiment 3)
cases = [("scene_007", "(a) correct accept", "valid grasp, accepted", "#1a9641", None),
         ("scene_204", "(b) false accept", "invalid grasp, accepted", "#d7191c", None),
         ("scene_073", "(c) correct reject", "invalid grasp, rejected", "#1a9641", "suction_area"),
         ("scene_009", "(d) false reject", "valid grasp, rejected", "#c51b8a", "bbox_extent")]
fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
for ax, (sid, lab, sub, col, check) in zip(axes.ravel(), cases):
    rgb = plt.imread(DATA / sid / "rgb.png")
    preds, ws = _std(sid)
    u, v = _grasp_px(sid)
    ax.imshow(composite(rgb, preds))
    for m in preds:
        ax.contour(m, levels=[0.5], colors=["black"], linewidths=0.5)
    ax.scatter([u], [v], s=340, facecolors="none", edgecolors=col, linewidths=3.2)
    ax.scatter([u], [v], s=25, c=col, marker="x", linewidths=2.5)
    ax.set_title(f"{lab}: {sub}" + (f"\ndecisive check: {check}" if check else ""), fontsize=9.5)
    ys, xs = np.where(ws)
    ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.max(), ys.min())
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(OUT / "exp3_decision_examples.pdf", bbox_inches="tight", dpi=200)
print("saved intro/sam-failure/decision figures", OUT)
