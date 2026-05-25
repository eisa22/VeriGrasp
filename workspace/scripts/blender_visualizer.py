#!/usr/bin/env python3
"""RGB-D → PLY und Open3D-Viewer für Blender-Szenen unter Data/blender_data.

Einzelne Szene (wie gewohnt):
  python scripts/blender_visualizer.py --scene 0 --view

Alle Szenen gleich konvertieren + nacheinander ansehen:
  python scripts/blender_visualizer.py --run-all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import Imath
import open3d as o3d

FX = FY = 640.0
CX = 640.0
CY = 512.0
DEPTH_MIN = 0.5  # m — alles darunter = Background
DEPTH_MAX = 3.5  # m
CAM_Z_M = 3.272
SCENE_X_OFFSET_M = 1.5


def resolve_data_dir(start: Path | None = None) -> Path:
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start)
    candidates.extend(
        [
            Path.cwd() / "Data/blender_data/pallet_stack_output",
            here.parent / "Data/blender_data/pallet_stack_output",
        ]
    )
    for path in candidates:
        path = path.expanduser().resolve()
        if path.is_dir() and any(path.glob("scene_*")):
            return path
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Kein pallet_stack_output mit scene_* gefunden.\nVersucht:\n  {tried}"
    )


def load_depth(exr_path: Path) -> np.ndarray:
    f = OpenEXR.InputFile(str(exr_path))
    dw = f.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    raw = f.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
    return np.frombuffer(raw, dtype=np.float32).reshape(h, w)


def load_rgb(png_path: Path, h: int, w: int) -> np.ndarray:
    img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB nicht lesbar: {png_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    return img


def depth_to_pointcloud(depth: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mask = (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX)
    d = depth[mask]
    x = (uu[mask] - CX) * d / FX
    y = (vv[mask] - CY) * d / FY
    z = CAM_Z_M - d
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    return xyz, rgb[mask]


def write_ply(path: Path, xyz: np.ndarray, colors: np.ndarray) -> None:
    n = len(xyz)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    data = np.zeros(n, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["r"], data["g"], data["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    print(f"  → {n:,} Punkte gespeichert: {path.name}")


def process_scene(scene_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    depth = load_depth(scene_dir / "depth.exr")
    rgb = load_rgb(scene_dir / "rgb.png", *depth.shape)
    xyz, colors = depth_to_pointcloud(depth, rgb)
    write_ply(scene_dir / "pointcloud.ply", xyz, colors)
    return xyz, colors


def view_ply(ply_path: Path, window_name: str | None = None, index: int | None = None, total: int | None = None) -> None:
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY fehlt: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    base = window_name or ply_path.parent.name
    if index is not None and total is not None:
        title = f"{base} ({index + 1}/{total})"
    else:
        title = base
    print(f"Open3D: {title} — {len(pcd.points):,} Punkte (Fenster schließen → weiter)")
    o3d.visualization.draw_geometries([pcd], window_name=title)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blender RGB-D (depth.exr + rgb.png) → pointcloud.ply + Open3D-Viewer."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="pallet_stack_output (Standard: Data/blender_data/pallet_stack_output)",
    )
    parser.add_argument("--scene", type=int, default=None, help="Nur scene_XX (z.B. 0 → scene_00)")
    parser.add_argument("--merge", action="store_true", help="Zusätzlich all_scenes.ply erzeugen")
    parser.add_argument(
        "--view",
        action="store_true",
        help="Nach Konvertierung in Open3D anzeigen (wie draw_geometries)",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Nur vorhandene PLY anzeigen, keine Neuberechnung",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Alle 21 Szenen wie --scene 0 --view: konvertieren + nacheinander anzeigen",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Bei --run-all / --view-only: ab scene_XX starten (0 = scene_00)",
    )
    args = parser.parse_args()

    if args.run_all:
        args.view = True

    data_dir = resolve_data_dir(args.data)
    print(f"Daten: {data_dir}")

    if args.view_only:
        if args.merge:
            view_ply(data_dir / "all_scenes.ply", "all_scenes")
        else:
            all_indices = list(range(args.start, 21))
            for j, idx in enumerate(all_indices):
                view_ply(
                    data_dir / f"scene_{idx:02d}" / "pointcloud.ply",
                    f"scene_{idx:02d}",
                    index=j,
                    total=len(all_indices),
                )
        return 0

    if args.scene is not None and not args.run_all:
        scenes = [args.scene]
    else:
        scenes = list(range(args.start, 21))
    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []

    total = len(scenes)
    for j, i in enumerate(scenes):
        scene_dir = data_dir / f"scene_{i:02d}"
        if not scene_dir.is_dir():
            print(f"[Warnung] Übersprungen (fehlt): {scene_dir.name}", file=sys.stderr)
            continue
        print(f"\n[{j + 1}/{total}] scene_{i:02d}:")
        try:
            xyz, colors = process_scene(scene_dir)
        except Exception as exc:
            print(f"  FEHLER: {exc}", file=sys.stderr)
            return 1
        if args.merge:
            all_xyz.append(xyz + np.array([i * SCENE_X_OFFSET_M, 0, 0], dtype=np.float32))
            all_rgb.append(colors)
        if args.view and not args.merge:
            view_ply(
                scene_dir / "pointcloud.ply",
                f"scene_{i:02d}",
                index=j,
                total=total,
            )

    if args.merge:
        if not all_xyz:
            print("[Fehler] Keine Szene konvertiert.", file=sys.stderr)
            return 1
        print("\nMerged cloud:")
        merged_xyz = np.concatenate(all_xyz)
        merged_rgb = np.concatenate(all_rgb)
        write_ply(data_dir / "all_scenes.ply", merged_xyz, merged_rgb)
        if args.view:
            view_ply(data_dir / "all_scenes.ply", "all_scenes")

    print("\nFertig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
