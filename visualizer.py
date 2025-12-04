# visualize_single.py
import numpy as np
import open3d as o3d
from PIL import Image
import os

SESSION = "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_07"

RGB_PATH = os.path.join(SESSION, "rgb", "rgb_0000.png")
DEPTH_PATH = os.path.join(SESSION, "distance_to_image_plane", "distance_to_image_plane_0000.npy")


def load_pointcloud():

    # --- DEPTH LADEN ---
    depth = np.load(DEPTH_PATH)

    # --- RGB LADEN ---
    rgb_img = np.array(Image.open(RGB_PATH), dtype=np.uint8)

    if rgb_img.ndim != 3 or rgb_img.shape[2] < 3:
        raise ValueError("RGB-Bild hat falsche Dimensionen")

    rgb_np = rgb_img[:, :, :3].copy()

    # --- DEPTH → mm ---
    depth_mm = (depth * 1000).astype(np.uint16)

    # --- Open3D Images ---
    rgb_o3d = o3d.geometry.Image(rgb_np)
    depth_o3d = o3d.geometry.Image(depth_mm)

    # --- RGBD ---
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1000.0,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False
    )

    # --- Intrinsics ---
    h, w = depth.shape
    fx = fy = 437.04
    cx = w / 2
    cy = h / 2

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w, h, fx, fy, cx, cy
    )

    # --- Punktwolke aus RGBD ---
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # --- Orientierung korrigieren (wie in deinem funktionierenden Code) ---
    pcd.transform([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ])

    return pcd


def main():

    print("📂 Lade Punktwolke aus:", SESSION)

    pcd = load_pointcloud()

    # --- VISUALIZATION ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Single Session", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    print("🎉 Fertig! Punktwolke angezeigt.")


if __name__ == "__main__":
    main()
