#!/usr/bin/env python3
"""Convert Blender RGB-D renders (rgb.png + depth.exr) to PLY point clouds."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import open3d as o3d

# Pinhole intrinsics (1280×1024, 18 mm lens, 36 mm sensor)
FX = FY = 640.0
CX = 640.0
CY = 512.0
MAX_DEPTH_M = 3.5
SCENE_X_OFFSET_M = 1.5  # spacing when merging all scenes for visualization


def resolve_data_dir(start: Path | None = None) -> Path:
    here = Path(__file__).resolve().parent
    candidates = []
    if start is not None:
        candidates.append(start)
    candidates.extend(
        [
            Path.cwd() / "Data/blender_data/pallet_stack_output",
            here.parent / "Data/blender_data/pallet_stack_output",
            here.parents[2] / "blender_visualisation/pallet_stack_output",
        ]
    )
    for path in candidates:
        path = path.expanduser().resolve()
        if path.is_dir() and any(path.glob("scene_*")):
            return path
    tried = "\n  ".join(str(p.expanduser().resolve()) for p in candidates)
    raise FileNotFoundError(f"No pallet_stack_output with scene_* folders found.\nTried:\n  {tried}")


def load_depth_exr(path: Path) -> np.ndarray:
    """Load Z-depth (meters) from OpenEXR. Tries OpenEXR+Imath, then OpenCV, then imageio."""
    errors: list[str] = []

    try:
        import Imath
        import OpenEXR

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = list(header["channels"].keys())
        channel = next((c for c in ("Z", "R", "Y", "G", "B", "A") if c in channels), channels[0])
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = exr.channel(channel, pixel_type)
        depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        return depth
    except ImportError as exc:
        errors.append(f"OpenEXR+Imath: {exc}")
    except Exception as exc:
        errors.append(f"OpenEXR+Imath: {exc}")

    try:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError("cv2.imread returned None")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth.astype(np.float32)
    except Exception as exc:
        errors.append(f"OpenCV: {exc}")

    try:
        import imageio.v3 as iio

        depth = np.asarray(iio.imread(path), dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth
    except Exception as exc:
        errors.append(f"imageio: {exc}")

    raise RuntimeError(f"Failed to read {path}:\n  " + "\n  ".join(errors))


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def depth_rgb_to_pointcloud(depth: np.ndarray, rgb: np.ndarray) -> o3d.geometry.PointCloud:
    if depth.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"Depth {depth.shape[:2]} and RGB {rgb.shape[:2]} size mismatch")

    height, width = depth.shape[:2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    z = depth.astype(np.float32)
    valid = (z > 0.0) & (z <= MAX_DEPTH_M) & np.isfinite(z)

    x = (u - CX) * z / FX
    y = (v - CY) * z / FY

    points = np.ascontiguousarray(
        np.stack([x[valid], y[valid], z[valid]], axis=-1), dtype=np.float64
    )
    colors = np.ascontiguousarray(rgb[valid], dtype=np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def convert_scene(scene_dir: Path, write_ascii: bool = False) -> o3d.geometry.PointCloud:
    depth_path = scene_dir / "depth.exr"
    rgb_path = scene_dir / "rgb.png"
    out_path = scene_dir / "pointcloud.ply"

    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth.exr in {scene_dir.name}")
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing rgb.png in {scene_dir.name}")

    depth = load_depth_exr(depth_path)
    rgb = load_rgb(rgb_path)
    pcd = depth_rgb_to_pointcloud(depth, rgb)

    o3d.io.write_point_cloud(
        str(out_path),
        pcd,
        write_ascii=write_ascii,
        compressed=not write_ascii,
    )
    return pcd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Back-project Blender RGB-D scenes to pointcloud.ply files."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to pallet_stack_output (default: auto-detect Data/blender_data)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write ASCII PLY instead of binary (default: binary)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip writing merged all_scenes.ply",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=SCENE_X_OFFSET_M,
        help=f"X offset between scenes in all_scenes.ply (default: {SCENE_X_OFFSET_M} m)",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data)
    scenes = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("scene_"))
    if not scenes:
        print(f"[Error] No scene_* folders in {data_dir}", file=sys.stderr)
        return 1

    print(f"Data directory: {data_dir}")
    print(f"Scenes: {len(scenes)} | intrinsics fx=fy={FX}, cx={CX}, cy={CY}")
    print(f"Depth filter: 0 < Z <= {MAX_DEPTH_M} m\n")

    merged_clouds: list[o3d.geometry.PointCloud] = []

    for idx, scene_dir in enumerate(scenes):
        print(f"[{idx + 1:02d}/{len(scenes)}] {scene_dir.name} ...", end=" ", flush=True)
        try:
            pcd = convert_scene(scene_dir, write_ascii=args.ascii)
            n_points = len(pcd.points)
            print(f"{n_points:,} points -> {scene_dir / 'pointcloud.ply'}")

            if not args.no_merge:
                offset_pcd = o3d.geometry.PointCloud(pcd)
                points = np.asarray(offset_pcd.points)
                points[:, 0] += idx * args.offset
                offset_pcd.points = o3d.utility.Vector3dVector(points)
                merged_clouds.append(offset_pcd)
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1

    if not args.no_merge and merged_clouds:
        merged = merged_clouds[0]
        for cloud in merged_clouds[1:]:
            merged += cloud
        merged_path = data_dir / "all_scenes.ply"
        o3d.io.write_point_cloud(
            str(merged_path),
            merged,
            write_ascii=args.ascii,
            compressed=not args.ascii,
        )
        print(f"\nMerged cloud: {len(merged.points):,} points -> {merged_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
