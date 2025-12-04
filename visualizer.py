# sam3d.py
import numpy as np
import open3d as o3d
from PIL import Image
import os
from config import DEBUG


class SAM3D:
    """
    SAM3D erzeugt pro Maske eine echte 3D-Punktwolke auf Basis von:
    - RGB
    - Depth
    - Kamera-Intrinsics
    """

    def __init__(self, session_path):
        self.session = session_path

        # Dateipfade
        self.rgb_path = os.path.join(session_path, "rgb", "rgb_0000.png")
        self.depth_path = os.path.join(session_path, "distance_to_image_plane",
                                       "distance_to_image_plane_0000.npy")

        # Daten laden
        self.rgb = np.array(Image.open(self.rgb_path), dtype=np.uint8)
        self.depth = np.load(self.depth_path)

        # Dimensionscheck
        if self.rgb.shape[:2] != self.depth.shape[:2]:
            raise ValueError("RGB und Depth haben unterschiedliche Größe!")

        self.H, self.W = self.depth.shape

        # Intrinsics (aus funktionierendem Code)
        self.fx = self.fy = 437.04
        self.cx = self.W / 2
        self.cy = self.H / 2

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.W, self.H, self.fx, self.fy, self.cx, self.cy
        )

        if DEBUG:
            print(f"[SAM3D] Session: {self.session}")
            print(f"[SAM3D] Auflösung: {self.W}x{self.H}")
            print(f"[SAM3D] Intrinsics fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    # ---------------------------------------------------------
    # Punktwolke erzeugen (einmal pro Frame)
    # ---------------------------------------------------------
    def _create_pointcloud(self):
        rgb_o3d = o3d.geometry.Image(self.rgb[:, :, :3])
        depth_mm = (self.depth * 1000).astype(np.uint16)
        depth_o3d = o3d.geometry.Image(depth_mm)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)

        # Orientation Fix
        pcd.transform([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])

        return pcd

    # ---------------------------------------------------------
    # Masken → Punktwolken
    # ---------------------------------------------------------
    def process(self, masks):

        full_pcd = self._create_pointcloud()
        points = np.asarray(full_pcd.points)
        colors = np.asarray(full_pcd.colors)

        pcs = []

        for idx, mask in enumerate(masks):

            ys, xs = np.where(mask > 0)

            # Leere Masken komplett überspringen
            if len(xs) == 0:
                if DEBUG:
                    print(f"[SAM3D] Maske {idx} enthält keine Pixel – übersprungen.")
                continue

            linear_idx = ys * self.W + xs

            obj_points = points[linear_idx]
            obj_colors = colors[linear_idx]

            pcs.append((obj_points, obj_colors))

            if DEBUG:
                print(f"[SAM3D] Objekt {idx}: {obj_points.shape[0]} Punkte")

        if DEBUG and len(pcs) > 0:
            self._visualize(pcs)

        return pcs

    # ---------------------------------------------------------
    # Visualisierung
    # ---------------------------------------------------------
    def _visualize(self, pcs):

        geoms = []

        for (pts, cols) in pcs:

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            geoms.append(pcd)

        print("[SAM3D] Öffne Open3D Viewer…")
        o3d.visualization.draw_geometries(geoms)
