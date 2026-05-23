#!/usr/bin/env python3
"""2D-Overlay: RGB + Stage-1-Masken (garantiert sichtbar, kein Open3D)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from perception.viz.overlay import render_frame
from perception.viz.pointcloud import (
    list_frames,
    load_predictions_grouped,
    load_rgb_depth,
    predictions_to_masks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D mask overlay viewer")
    parser.add_argument("--dataset", default="Data/unified_rgbd")
    parser.add_argument("--run", default="runs/unified_classical")
    parser.add_argument("--frame", type=int, default=0, help="Frame index 0..N-1")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    run_dir = Path(args.run)
    frames = list_frames(dataset)
    grouped = load_predictions_grouped(run_dir, dataset)

    idx = args.frame % len(frames)
    fr = frames[idx]
    preds = grouped.get(fr.image_id, [])
    rgb, depth, _ = load_rgb_depth(fr.rgb_path, fr.depth_path)
    masks = predictions_to_masks(preds, depth.shape)

    vis = render_frame(rgb, preds, alpha=0.55)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    if masks:
        combined = np.zeros(rgb.shape[:2], dtype=np.int32)
        for i, m in enumerate(masks):
            combined[m > 0] = i + 1
        axes[1].imshow(combined, cmap="tab10")
        axes[1].set_title(f"Masken ({len(masks)} Inst.)")
    else:
        axes[1].text(0.5, 0.5, "KEINE MASKEN", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Masken")
    axes[1].axis("off")

    axes[2].imshow(vis)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(f"{fr.label} | image_id={fr.image_id} | {len(preds)} predictions")
    plt.tight_layout()
    print(f"Frame {idx+1}/{len(frames)}: {fr.label}, {len(preds)} Masken")
    print("Pfeiltasten: nicht unterstützt — --frame N für anderen Frame")
    plt.show()


if __name__ == "__main__":
    main()
