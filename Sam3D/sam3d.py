# sam3d.py

import numpy as np
from config import DEBUG
from path_utils import get_pointcloud_path

import open3d as o3d  # <---- NEW

class SAM3D:

    def __init__(self):
        self.pc = np.load(get_pointcloud_path())

        # --------- Case A: (H, W, 3) → already XYZ grid ---------
        if self.pc.ndim == 3 and self.pc.shape[2] == 3:
            self.mode = "xyz_raster"
            self.H, self.W, _ = self.pc.shape

        # --------- Case B: (N, 3) → flatten XYZ → reshape to (H,W,3) ---------
        elif self.pc.ndim == 2 and self.pc.shape[1] == 3:
            N = self.pc.shape[0]
            side = int(np.sqrt(N))

            if side * side != N:
                raise ValueError(f"Cannot reshape pointcloud of shape {self.pc.shape} to square grid")

            self.pc = self.pc.reshape(side, side, 3)
            self.mode = "xyz_flat_to_raster"
            self.H, self.W, _ = self.pc.shape

        else:
            raise ValueError(f"Unsupported pointcloud shape: {self.pc.shape}")

        if DEBUG:
            print(f"[SAM3D] Loaded mode={self.mode}, reshaped to {self.pc.shape}")


    def process(self, masks):
        pcs = []

        for i, mask in enumerate(masks):
            ys, xs = np.where(mask > 0)
            xyz = self.pc[ys, xs]

            pcs.append(xyz)

            if DEBUG:
                print(f"[SAM3D] Objekt {i}: {xyz.shape[0]} Punkte")

        if DEBUG:
            self.visualize(pcs)

        return pcs


    def visualize(self, pcs):
        """ Visualize multiple segmented pointclouds in Open3D with random colors. """

        geometries = []
        rng = np.random.default_rng(42)

        for pc in pcs:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)

            # random color per object
            color = rng.random(3)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(color, (pc.shape[0], 1))
            )

            geometries.append(pcd)

        print("[SAM3D] Öffne 3D Visualisierung... (Fenster schließt sich nach ESC)")

        o3d.visualization.draw_geometries(geometries)
