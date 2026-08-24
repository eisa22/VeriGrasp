"""Regenerates the per-scene pointcloud.ply files of SynDePal.

The point clouds are derivable data: they follow from depth.npy and the
camera intrinsics recorded in Data/blender_dataset/dataset_meta.json
(pinhole model, OpenCV camera frame: x right, y down, z forward). They
are therefore not versioned in this repository; run this script once
after cloning if you need them.

Usage (from the repository root):
    python scripts/regenerate_pointclouds.py [--data-root Data/blender_dataset]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "Data/blender_dataset")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rewrite pointcloud.ply even where it exists")
    args = ap.parse_args()

    meta = json.load(open(args.data_root / "dataset_meta.json"))
    cam = meta["camera"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    scenes = sorted(p for p in args.data_root.iterdir() if p.name.startswith("scene_"))
    n_done = 0
    for sp in scenes:
        out = sp / "pointcloud.ply"
        if out.exists() and not args.overwrite:
            continue
        depth = np.load(sp / "depth.npy").astype(np.float64)
        H, W = depth.shape
        # pixel centres (+0.5), matching the convention of the original
        # point clouds (back-projection lands on u in [0.5, W-0.5])
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        z = depth
        valid = z > 0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        rgb_path = sp / "rgb.png"
        if rgb_path.exists():
            import matplotlib.image as mpimg
            rgb = mpimg.imread(rgb_path)[..., :3]
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            pcd.colors = o3d.utility.Vector3dVector(rgb[valid])
        o3d.io.write_point_cloud(str(out), pcd)
        n_done += 1
        if n_done % 100 == 0:
            print(f"{n_done} regenerated...", flush=True)
    print(f"done: {n_done} point clouds written under {args.data_root}")


if __name__ == "__main__":
    main()
