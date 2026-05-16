"""
PLY Utilities - Direkte Verarbeitung von PLY-Dateien.
Ermöglicht der Pipeline, Original-PLY-Dateien ohne Depth-Konvertierung zu nutzen.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image


def load_pointcloud_from_ply(ply_path, rgb_path=None):
    """
    Lädt Punktwolke direkt aus PLY-Datei.
    
    Args:
        ply_path: Pfad zur PLY-Datei
        rgb_path: Optional, falls RGB separat geladen werden soll
    
    Returns:
        points: (N, 3) numpy array - 3D Punkte
        colors: (N, 3) numpy array - RGB Farben (0-1 float)
        rgb_image: PIL Image falls rgb_path gegeben
    """
    ply_path = Path(ply_path)
    
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY nicht gefunden: {ply_path}")
    
    # Lade PLY mit Open3D
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if len(points) == 0:
        raise ValueError(f"PLY enthält keine Punkte: {ply_path}")
    
    # RGB-Bild laden falls angegeben
    rgb_image = None
    if rgb_path:
        rgb_image = Image.open(rgb_path)
    
    return points, colors, rgb_image


def extract_depth_from_ply(ply_path, rgb_path, fx=437.04, fy=437.04):
    """
    Extrahiert Depth-Map aus PLY-Datei (on-the-fly).
    
    Args:
        ply_path: Pfad zur PLY-Datei
        rgb_path: Pfad zum RGB-Bild (für Dimensionen)
        fx, fy: Kamera Focal Length
    
    Returns:
        depth_map: (H, W) numpy array mit Tiefenwerten in Metern
    """
    # Lade PLY
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        raise ValueError(f"PLY enthält keine Punkte: {ply_path}")
    
    # RGB für Dimensionen
    rgb_img = Image.open(rgb_path)
    width, height = rgb_img.size
    cx, cy = width / 2, height / 2
    
    # PLY-Koordinaten: Z ist negativ (Szene in -Z Richtung)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    depth_values = -z  # Tiefe = -Z
    
    # Filtere invalide Punkte
    valid_mask = (depth_values > 0) & (depth_values < 10.0)
    x = x[valid_mask]
    y = y[valid_mask]
    depth_values = depth_values[valid_mask]
    
    # Initialisiere Depth-Map
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    # Projiziere Punkte auf 2D mit Z-Buffer
    for i in range(len(x)):
        u = int(x[i] * fx / depth_values[i] + cx)
        v = int(y[i] * fy / depth_values[i] + cy)
        
        if 0 <= u < width and 0 <= v < height:
            # Z-Buffer: Behalte näheren Punkt
            if depth_map[v, u] == 0 or depth_values[i] < depth_map[v, u]:
                depth_map[v, u] = depth_values[i]
    
    return depth_map


def create_rgbd_from_ply(ply_path, rgb_path, fx=437.04, fy=437.04):
    """
    Erstellt RGBD-Daten direkt aus PLY + RGB.
    Verwendet RGB-Bild für Farben (nicht PLY-Farben).
    
    Args:
        ply_path: Pfad zur PLY-Datei
        rgb_path: Pfad zum RGB-Bild
        fx, fy: Kamera Focal Length
    
    Returns:
        all_points: (N, 3) 3D-Punkte der gesamten Szene
        rgb: (H, W, 3) RGB-Bild als numpy array
        H, W: Bilddimensionen
    """
    # Lade PLY (nur Geometrie, keine Farben)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        raise ValueError(f"PLY enthält keine Punkte: {ply_path}")
    
    # Lade RGB-Bild (verwende DIESES für Farben, nicht PLY)
    rgb = np.array(Image.open(rgb_path))[:, :, :3]
    H, W = rgb.shape[:2]
    
    # Transformiere Koordinaten für Open3D-Visualisierung
    # PLY: Z negativ (Szene in -Z) → Invertiere für Visualisierung
    transformed_points = points.copy()
    transformed_points[:, 2] = -transformed_points[:, 2]  # Z invertieren
    transformed_points[:, 1] = -transformed_points[:, 1]  # Y invertieren (Open3D Convention)
    
    return transformed_points, rgb, H, W


def ply_to_open3d_pointcloud(ply_path):
    """
    Lädt PLY und gibt Open3D PointCloud Objekt zurück.
    Korrekt transformiert für Visualisierung.
    
    Args:
        ply_path: Pfad zur PLY-Datei
    
    Returns:
        o3d.geometry.PointCloud
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    # Transformiere für korrekte Visualisierung
    points = np.asarray(pcd.points)
    points[:, 2] = -points[:, 2]  # Z invertieren
    points[:, 1] = -points[:, 1]  # Y invertieren
    
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd


def assign_rgb_colors_to_points(points, rgb_image, H, W, fx=437.04, fy=437.04):
    """
    Ordnet RGB-Farben den 3D-Punkten durch Projektion zu.
    
    Args:
        points: (N, 3) transformierte 3D-Punkte (bereits für Open3D)
        rgb_image: (H, W, 3) RGB-Bild
        H, W: Bilddimensionen
        fx, fy: Kamera Focal Length
    
    Returns:
        colors: (N, 3) RGB-Farben (0-1 normalisiert)
    """
    cx, cy = W / 2, H / 2
    
    # Rück-transformiere Punkte (Open3D → Kamera-Koordinaten)
    # Open3D hat Y und Z invertiert
    cam_points = points.copy()
    cam_points[:, 1] = -cam_points[:, 1]  # Y zurück
    cam_points[:, 2] = -cam_points[:, 2]  # Z zurück (ist jetzt positiv)
    
    # Projiziere auf 2D
    z = cam_points[:, 2]
    z[z <= 0] = 0.01  # Verhindere Division durch 0
    
    u = (cam_points[:, 0] * fx / z + cx).astype(int)
    v = (cam_points[:, 1] * fy / z + cy).astype(int)
    
    # Bounds check und Farben zuordnen
    colors = np.ones((len(points), 3)) * 0.5  # Default: grau
    
    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    
    u_valid = np.clip(u[valid_mask], 0, W - 1)
    v_valid = np.clip(v[valid_mask], 0, H - 1)
    
    colors[valid_mask] = rgb_image[v_valid, u_valid] / 255.0
    
    return colors


if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) > 1:
        ply_path = sys.argv[1]
        print(f"Teste PLY: {ply_path}")
        
        points, colors, _ = load_pointcloud_from_ply(ply_path)
        print(f"✓ Punkte: {len(points)}")
        print(f"✓ Farben: {len(colors)}")
        print(f"✓ Z-Range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]m")
    else:
        print("Usage: python ply_utils.py <ply_path>")
