# Sam3D/sam3d.py
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

    Die Punktwolke wird über Open3D aus einem RGBD-Image erzeugt
    (identisch zu deinem funktionierenden Visualizer).
    """

    def __init__(self, session_path: str):
        self.session = session_path

        # Dateipfade
        self.rgb_path = os.path.join(session_path, "rgb", "rgb_0000.png")
        self.depth_path = os.path.join(
            session_path,
            "distance_to_image_plane",
            "distance_to_image_plane_0000.npy",
        )

        # -------------------------
        # Daten laden (contiguous!)
        # -------------------------
        rgb_img = np.array(Image.open(self.rgb_path), dtype=np.uint8)
        self.rgb = np.ascontiguousarray(rgb_img[:, :, :3])  # sicherstellen: H×W×3

        depth_np = np.load(self.depth_path)
        self.depth = np.ascontiguousarray(depth_np)

        # Dimensionscheck
        if self.rgb.shape[:2] != self.depth.shape[:2]:
            raise ValueError(
                f"RGB und Depth haben unterschiedliche Größe: "
                f"RGB={self.rgb.shape[:2]}, Depth={self.depth.shape[:2]}"
            )

        self.H, self.W = self.depth.shape

        # Intrinsics (aus deinem funktionierenden Code)
        self.fx = self.fy = 437.04
        self.cx = self.W / 2
        self.cy = self.H / 2

        # Open3D Intrinsic Objekt
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.W, self.H, self.fx, self.fy, self.cx, self.cy
        )

        if DEBUG:
            print(f"[SAM3D] Session: {self.session}")
            print(f"[SAM3D] Auflösung: {self.W}x{self.H}")
            print(
                f"[SAM3D] Intrinsics fx={self.fx}, fy={self.fy}, "
                f"cx={self.cx}, cy={self.cy}"
            )

    # ---------------------------------------------------------
    # Punktwolke erzeugen (einmal pro Frame)
    # ---------------------------------------------------------
    def _create_pointcloud(self) -> o3d.geometry.PointCloud:
        # Images für Open3D: müssen contiguous sein
        rgb_o3d = o3d.geometry.Image(self.rgb)  # H×W×3, uint8, contiguous

        depth_mm = np.ascontiguousarray((self.depth * 1000.0).astype(np.uint16))
        depth_o3d = o3d.geometry.Image(depth_mm)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            self.intrinsic,
        )

        # Orientierung wie in deinem funktionierenden Code
        pcd.transform(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )

        return pcd

    # ---------------------------------------------------------
    # Masken → Punktwolken
    # masks: Liste von 2D-Arrays (H,W), Werte {0,1} oder bool
    # ---------------------------------------------------------
    def process(self, masks):
        # Vollständige Punktwolke nur EINMAL erzeugen
        full_pcd = self._create_pointcloud()
        points = np.asarray(full_pcd.points)
        colors = np.asarray(full_pcd.colors)

        pcs = []

        for idx, mask in enumerate(masks):
            mask_np = np.asarray(mask)

            if mask_np.shape != (self.H, self.W):
                raise ValueError(
                    f"Maske {idx} hat falsche Shape: {mask_np.shape}, erwartet {(self.H, self.W)}"
                )

            # Pixel, die zur Maske gehören
            ys, xs = np.where(mask_np > 0)

            # Leere Masken überspringen
            if len(xs) == 0:
                if DEBUG:
                    print(f"[SAM3D] Maske {idx} enthält keine Pixel – übersprungen.")
                continue

            # Pixel (y,x) → linearer Index in der Punktwolke
            linear_idx = ys * self.W + xs

            obj_points = points[linear_idx]
            obj_colors = colors[linear_idx]

            pcs.append((obj_points, obj_colors))

            if DEBUG:
                print(f"[SAM3D] Objekt {idx}: {obj_points.shape[0]} Punkte")

        # Optional 3D-Visualisierung
        if DEBUG and len(pcs) > 0:
            self._visualize(pcs)

        return pcs

    # ---------------------------------------------------------
    # Visualisierung
    # ---------------------------------------------------------
    def _visualize(self, pcs):
        print("[SAM3D] Visualisiere vollständige Szene …")

        # --- 1) Gesamte Punktwolke erzeugen ---
        full_pcd = self._create_pointcloud()

        # komplette Szene in hellgrau färben
        gray = np.full((len(full_pcd.points), 3), 0.7)
        full_pcd.colors = o3d.utility.Vector3dVector(gray)

        geoms = [full_pcd]

        # --- 2) Jedes Objekt farbig hervorheben ---
        rng = np.random.default_rng()

        for i, (pts, cols) in enumerate(pcs):

            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(pts)

            # zufällige Farbe für Objekt
            random_color = rng.uniform(0.2, 1.0, size=3)
            highlight = np.tile(random_color, (pts.shape[0], 1))
            pcd_obj.colors = o3d.utility.Vector3dVector(highlight)

            geoms.append(pcd_obj)

        print("[SAM3D] Öffne Open3D Viewer …")
        o3d.visualization.draw_geometries(geoms)