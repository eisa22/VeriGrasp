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
        print(f"\n{'='*60}")
        print(f"SAM3D - 3D Segmentierung")
        print(f"{'='*60}")
        print(f"Eingabe: {len(masks)} 2D-Masken von SAM")
        print(f"Ausgabe: 3D-Punktwolken für jedes Objekt")
        print(f"")
        
        # Vollständige Punktwolke nur EINMAL erzeugen
        full_pcd = self._create_pointcloud()
        points = np.asarray(full_pcd.points)
        colors = np.asarray(full_pcd.colors)

        # Erstelle eine Zuordnungsmaske: Welcher Pixel gehört zu welchem Objekt?
        # -1 = kein Objekt, sonst Index des Objekts
        assignment_map = np.full((self.H, self.W), -1, dtype=np.int32)

        pcs = []

        for idx, mask in enumerate(masks):
            mask_np = np.asarray(mask)
            non_zero = np.count_nonzero(mask_np)
            print(f"Maske {idx}: {non_zero} Pixel → ", end="")

            if mask_np.shape != (self.H, self.W):
                raise ValueError(
                    f"Maske {idx} hat falsche Shape: {mask_np.shape}, erwartet {(self.H, self.W)}"
                )

            # Pixel, die zur Maske gehören
            ys, xs = np.where(mask_np > 0)

            # Leere Masken überspringen
            if len(xs) == 0:
                print(f"leer, übersprungen")
                continue

            # WICHTIG: Nur Pixel verwenden, die noch KEINEM anderen Objekt zugeordnet sind
            # Dies verhindert Überlappungen (first-come-first-served)
            unassigned_mask = assignment_map[ys, xs] == -1
            ys_unique = ys[unassigned_mask]
            xs_unique = xs[unassigned_mask]

            if len(xs_unique) == 0:
                print(f"alle Pixel schon zugeordnet, übersprungen")
                continue

            # Markiere diese Pixel als diesem Objekt zugehörig
            assignment_map[ys_unique, xs_unique] = idx

            # Pixel (y,x) → linearer Index in der Punktwolke
            linear_idx = ys_unique * self.W + xs_unique

            obj_points = points[linear_idx]
            obj_colors = colors[linear_idx]

            pcs.append((obj_points, obj_colors))

            overlapping = len(ys) - len(ys_unique)
            print(f"{obj_points.shape[0]} 3D-Punkte ({overlapping} überlappend)")

        print(f"\n{'='*60}")
        print(f"SAM3D Zusammenfassung")
        print(f"{'='*60}")
        print(f"Eingabe: {len(masks)} 2D-Masken")
        print(f"Ausgabe: {len(pcs)} 3D-Punktwolken")
        print(f"{'='*60}\n")

        # Optional 3D-Visualisierung
        if DEBUG and len(pcs) > 0:
            self._visualize(pcs)

        return pcs

    # ---------------------------------------------------------
    # Visualisierung
    # ---------------------------------------------------------
    def _visualize(self, pcs):
        print(f"{'='*60}")
        print(f"3D-Visualisierung")
        print(f"{'='*60}")
        print(f"Zeige {len(pcs)} segmentierte Objekte in 3D")

        # --- 1) Gesamte Punktwolke erzeugen ---
        full_pcd = self._create_pointcloud()

        # komplette Szene in hellgrau färben
        gray = np.full((len(full_pcd.points), 3), 0.7)
        full_pcd.colors = o3d.utility.Vector3dVector(gray)

        geoms = [full_pcd]

        # --- 2) Jedes Objekt farbig hervorheben ---
        # Vordefinierte, gut unterscheidbare Farben
        distinct_colors = [
            [1.0, 0.0, 0.0],  # Rot
            [0.0, 1.0, 0.0],  # Grün
            [0.0, 0.0, 1.0],  # Blau
            [1.0, 1.0, 0.0],  # Gelb
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Violett
        ]

        for i, (pts, cols) in enumerate(pcs):
            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(pts)

            # Verwende vordefinierte Farben (zyklisch wiederholen wenn mehr Objekte)
            color = distinct_colors[i % len(distinct_colors)]
            color_names = ["Rot", "Grün", "Blau", "Gelb", "Magenta", "Cyan", "Orange", "Violett"]
            print(f"  Objekt {i}: {pts.shape[0]} Punkte → {color_names[i % len(color_names)]}")
            
            highlight = np.tile(color, (pts.shape[0], 1))
            pcd_obj.colors = o3d.utility.Vector3dVector(highlight)

            geoms.append(pcd_obj)

        print(f"\n→ Öffne Open3D Viewer...")
        print(f"  (Schließen Sie das Fenster um fortzufahren)\n")
        o3d.visualization.draw_geometries(geoms)