#!/usr/bin/env python3
"""Convert Blender RGB-D renders (rgb.png + depth.exr) to PLY point clouds."""

from __future__ import annotations

import argparse
import json
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
MIN_DEPTH_M = 0.5   # drop sky/background (depth ≈ 0 in Blender G channel)
MAX_DEPTH_M = 3.4   # drop noise beyond the floor (cam_z + small tolerance)
SCENE_X_OFFSET_M = 1.5  # spacing when merging all scenes for visualization
DEFAULT_CAM_Z_M = 3.272
DEFAULT_PALLET_SIZE = [0.801, 1.2, 0.144]
DEFAULT_PALLET_CENTER = [0.0, 0.0, 0.072]
SENSOR_WIDTH_MM = 36.0
VOXEL_SIZE_M = 0.004  # voxel downsample for cleaner PLY (~5 mm)


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


def _pick_depth_channel(channels_data: dict[str, np.ndarray], cam_z: float) -> tuple[str, np.ndarray]:
    """Auto-pick the depth channel whose VALID values lie in the physically plausible range.

    The Blender compositor stores three different depth encodings in R/G/B; the
    true camera-axis depth must satisfy ``MIN_DEPTH_M <= depth <= cam_z + eps``.
    We pick the channel with the **most** pixels in that band. If the auto-pick
    is ambiguous (e.g. all channels score similarly), we prefer "G" (cv2 idx 1)
    which is the canonical channel for this dataset.
    """
    low, high = MIN_DEPTH_M, cam_z + 0.3
    scores = {name: int(((arr >= low) & (arr <= high)).sum()) for name, arr in channels_data.items()}
    best_name = max(scores, key=scores.get)
    # If multiple channels tie within 5 %, prefer "G".
    if "G" in scores and scores["G"] >= 0.95 * scores[best_name]:
        best_name = "G"
    return best_name, channels_data[best_name]


def load_depth_exr(path: Path, cam_z: float = DEFAULT_CAM_Z_M) -> np.ndarray:
    """Load camera-axis depth (meters) from OpenEXR.

    The Blender export stores 3-4 channels (R, G, B, [A]); empirically the
    camera-axis depth lives in **G** (cv2 index 1) for the original render and
    in **R** when a depth material override is used. ``_pick_depth_channel``
    auto-selects the right one. Falls back to imageio if OpenEXR is missing.
    """
    errors: list[str] = []

    try:
        import Imath
        import OpenEXR

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        names = list(header["channels"].keys())
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        data = {
            name: np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32).reshape(height, width)
            for name in names
        }
        # Prefer named Z/Depth channels if present
        for preferred in ("Z", "Depth", "depth.Z", "depth"):
            if preferred in data:
                return data[preferred]
        _, depth = _pick_depth_channel(data, cam_z)
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
            n_ch = depth.shape[2]
            # OpenCV reads EXR as BGR(A) → indices 0=B, 1=G, 2=R
            channels = {"B": depth[:, :, 0], "G": depth[:, :, 1]}
            if n_ch >= 3:
                channels["R"] = depth[:, :, 2]
            _, depth = _pick_depth_channel(channels, cam_z)
        return depth.astype(np.float32)
    except Exception as exc:
        errors.append(f"OpenCV: {exc}")

    try:
        import imageio.v3 as iio

        depth = np.asarray(iio.imread(path), dtype=np.float32)
        if depth.ndim == 3:
            channels = {f"ch{i}": depth[:, :, i] for i in range(depth.shape[2])}
            _, depth = _pick_depth_channel(channels, cam_z)
        return depth
    except Exception as exc:
        errors.append(f"imageio: {exc}")

    raise RuntimeError(f"Failed to read {path}:\n  " + "\n  ".join(errors))


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _box_mesh_at_center(size: list[float], center: list[float]) -> o3d.geometry.TriangleMesh:
    sx, sy, sz = size
    cx, cy, cz = center
    origin = np.array([cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0], dtype=np.float64)
    mesh = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) + origin)
    mesh.compute_vertex_normals()
    return mesh


def load_pallet_dims(scene_dir: Path, data_dir: Path) -> tuple[list[float], list[float]]:
    for path in (scene_dir / "ground_truth.json", data_dir / "ground_truth.json"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            pallet = json.load(f).get("pallet", {})
        size = pallet.get("size_m") or pallet.get("size")
        center = pallet.get("center_xyz") or pallet.get("center")
        if size and center:
            return list(size), list(center)
    return DEFAULT_PALLET_SIZE, DEFAULT_PALLET_CENTER


def pointcloud_from_raycast(scene_dir: Path, data_dir: Path) -> o3d.geometry.PointCloud:
    """Build colored point cloud by raycasting GT meshes (reliable geometry)."""
    gt_path = scene_dir / "ground_truth.json"
    rgb_path = scene_dir / "rgb.png"
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    cam = gt["camera"]
    cam_z = float(cam["position"][2])
    lens_mm = float(cam["lens_mm"])
    w, h = int(cam["resolution_px"][0]), int(cam["resolution_px"][1])
    fx = fy = (lens_mm / SENSOR_WIDTH_MM) * w
    cx, cy = w / 2.0, h / 2.0

    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"Missing rgb.png in {scene_dir.name}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pallet_size, pallet_center = load_pallet_dims(scene_dir, data_dir)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(_box_mesh_at_center(pallet_size, pallet_center))
    )
    for box in gt.get("boxes", []):
        ray_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(_box_mesh_at_center(box["size"], box["center"]))
        )

    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    t_w_to_c = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, cam_z], [0, 0, 0, 1]], dtype=np.float32
    )
    rays = ray_scene.create_rays_pinhole(o3d.core.Tensor(k), o3d.core.Tensor(t_w_to_c), w, h)
    hits = ray_scene.cast_rays(rays)
    t_hit = hits["t_hit"].numpy()
    geom_ids = hits["geometry_ids"].numpy()

    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    scale = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    z_cam = t_hit / scale
    x_w = (u - cx) * z_cam / fx
    y_w = -(v - cy) * z_cam / fy
    z_w = cam_z - z_cam

    invalid = o3d.t.geometry.RaycastingScene.INVALID_ID
    valid = (geom_ids != invalid) & np.isfinite(t_hit)

    points = np.ascontiguousarray(
        np.stack([x_w[valid], y_w[valid], z_w[valid]], axis=-1), dtype=np.float64
    )
    colors = np.ascontiguousarray(rgb[valid], dtype=np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def load_camera_height_m(scene_dir: Path) -> float:
    gt_path = scene_dir / "ground_truth.json"
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            cam_z = float(json.load(f)["camera"]["position"][2])
        return cam_z
    return DEFAULT_CAM_Z_M


def depth_rgb_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    cam_z: float,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
) -> o3d.geometry.PointCloud:
    """Back-project camera-axis depth to world XYZ.

    Camera at (0, 0, cam_z), looking straight down (-Z), image +X aligned with
    world +X, image +V flipped to world +Y. ``depth`` is the camera-axis
    distance to the hit point (Blender G channel).
    """
    if depth.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"Depth {depth.shape[:2]} and RGB {rgb.shape[:2]} size mismatch")

    height, width = depth.shape[:2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    z_cam = depth.astype(np.float32)
    valid = (
        np.isfinite(z_cam)
        & (z_cam >= min_depth)
        & (z_cam <= max_depth)
    )

    # Pinhole back-projection (camera frame), then world transform.
    x_w = (u - CX) * z_cam / FX
    y_w = -(v - CY) * z_cam / FY
    z_w = cam_z - z_cam

    points = np.ascontiguousarray(
        np.stack([x_w[valid], y_w[valid], z_w[valid]], axis=-1), dtype=np.float64
    )
    colors = np.ascontiguousarray(rgb[valid], dtype=np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def convert_scene(
    scene_dir: Path,
    data_dir: Path,
    method: str = "raycast",
    write_ascii: bool = False,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
    voxel_size: float | None = VOXEL_SIZE_M,
) -> o3d.geometry.PointCloud:
    out_path = scene_dir / "pointcloud.ply"
    gt_path = scene_dir / "ground_truth.json"

    if method == "raycast":
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing ground_truth.json in {scene_dir.name}")
        pcd = pointcloud_from_raycast(scene_dir, data_dir)
    else:
        depth_path = scene_dir / "depth.exr"
        rgb_path = scene_dir / "rgb.png"
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth.exr in {scene_dir.name}")
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing rgb.png in {scene_dir.name}")
        cam_z = load_camera_height_m(scene_dir)
        depth = load_depth_exr(depth_path, cam_z=cam_z)
        rgb = load_rgb(rgb_path)
        pcd = depth_rgb_to_pointcloud(depth, rgb, cam_z, min_depth=min_depth, max_depth=max_depth)

    if voxel_size and len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

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
    parser.add_argument(
        "--min-depth",
        type=float,
        default=MIN_DEPTH_M,
        help=f"Drop pixels with depth below this (m). Default {MIN_DEPTH_M}. "
             "Lower this (e.g. 0.5) to keep tall/deformed boxes; raise it to drop edge-aliasing.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=MAX_DEPTH_M,
        help=f"Drop pixels with depth above this (m). Default {MAX_DEPTH_M}.",
    )
    parser.add_argument(
        "--method",
        choices=("raycast", "depth"),
        default="raycast",
        help="raycast = GT meshes + rgb (default, correct shape); depth = depth.exr back-projection",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=VOXEL_SIZE_M,
        help=f"Voxel downsample size in m (default {VOXEL_SIZE_M}, 0 = off)",
    )
    args = parser.parse_args()
    voxel = None if args.voxel <= 0 else args.voxel

    data_dir = resolve_data_dir(args.data)
    scenes = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("scene_"))
    if not scenes:
        print(f"[Error] No scene_* folders in {data_dir}", file=sys.stderr)
        return 1

    print(f"Data directory: {data_dir}")
    print(f"Scenes: {len(scenes)} | intrinsics fx=fy={FX}, cx={CX}, cy={CY}")
    if args.method == "raycast":
        print(f"Method: raycast (ground_truth meshes + rgb.png) | voxel={voxel or 'off'} m\n")
    else:
        print(
            f"Method: depth.exr | filter: {args.min_depth} <= depth <= {args.max_depth} m | "
            f"voxel={voxel or 'off'} m\n"
        )

    merged_clouds: list[o3d.geometry.PointCloud] = []

    for idx, scene_dir in enumerate(scenes):
        print(f"[{idx + 1:02d}/{len(scenes)}] {scene_dir.name} ...", end=" ", flush=True)
        try:
            pcd = convert_scene(
                scene_dir,
                data_dir,
                method=args.method,
                write_ascii=args.ascii,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                voxel_size=voxel,
            )
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
