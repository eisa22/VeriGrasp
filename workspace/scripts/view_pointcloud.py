#!/usr/bin/env python3
"""View PLY point clouds in the browser (no Open3D GUI / GLFW required)."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from convert_pointclouds import resolve_data_dir

DEFAULT_MAX_POINTS = 80_000


def subsample_pcd(pcd: o3d.geometry.PointCloud, max_points: int, seed: int = 0) -> o3d.geometry.PointCloud:
    n = len(pcd.points)
    if n <= max_points:
        return pcd
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    out = o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)[idx]
    out.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64))
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)[idx]
        out.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, dtype=np.float64))
    return out


def pcd_to_figure(pcd: o3d.geometry.PointCloud, title: str) -> go.Figure:
    pts = np.asarray(pcd.points)
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
        color = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    else:
        color = "rgb(180,180,180)"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=color, opacity=0.9),
                hovertemplate="X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            bgcolor="rgb(20,20,24)",
        ),
        paper_bgcolor="rgb(20,20,24)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def list_scenes(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("scene_"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export pointcloud.ply to an interactive HTML file (browser viewer)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="pallet_stack_output directory (auto-detected by default)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scene_00",
        help="Scene folder name, e.g. scene_00 (default: scene_00)",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="View all_scenes.ply instead of a single scene",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=f"Max points in HTML plot (default: {DEFAULT_MAX_POINTS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: outputs/<scene>.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the HTML file in the default browser",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data)

    if args.merged:
        ply_path = data_dir / "all_scenes.ply"
        label = "all_scenes"
        if args.max_points == DEFAULT_MAX_POINTS:
            args.max_points = 120_000
    else:
        ply_path = data_dir / args.scene / "pointcloud.ply"
        label = args.scene

    if not ply_path.exists():
        print(f"[Error] Missing: {ply_path}", file=sys.stderr)
        print("Run first: python3 scripts/convert_pointclouds.py", file=sys.stderr)
        return 1

    print(f"Loading {ply_path} ...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_total = len(pcd.points)
    pcd = subsample_pcd(pcd, args.max_points)
    print(f"  {n_total:,} points -> showing {len(pcd.points):,} in browser")

    out_path = args.out
    if out_path is None:
        out_dir = Path.cwd() / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pointcloud_{label}.html"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = pcd_to_figure(pcd, title=f"{label} ({len(pcd.points):,} / {n_total:,} points)")
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved: {out_path.resolve()}")

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())

    scenes = list_scenes(data_dir)
    if not args.merged and scenes:
        print("\nOther scenes:")
        for s in scenes[:5]:
            print(f"  python3 scripts/view_pointcloud.py --scene {s.name} --open")
        if len(scenes) > 5:
            print(f"  ... ({len(scenes)} scenes total, scene_00 .. scene_20)")
        print("\nMerged:")
        print("  python3 scripts/view_pointcloud.py --merged --open")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
