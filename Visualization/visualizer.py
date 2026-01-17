"""
Visualization/visualizer.py
Modul für 2D und 3D Visualisierung von Segmentierungsergebnissen.
"""
import numpy as np
import torch
import open3d as o3d
import os
from PIL import Image, ImageDraw


def visualize_2d(orig_image, boxes, masks, labels, scores):
    """
    2D Visualisierung mit farbigen Boxen und Masken.
    
    Args:
        orig_image: PIL Image (Original RGB)
        boxes: Liste von Bounding Boxes [x0, y0, x1, y1]
        masks: Liste von binären Masken
        labels: Liste von Label-Strings
        scores: Liste von Confidence Scores
    """
    img = orig_image.copy()
    draw = ImageDraw.Draw(img)
    
    # Feste Farben für bessere Unterscheidbarkeit
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    ]
    
    for i, (box, mask, label, score) in enumerate(zip(boxes, masks, labels, scores)):
        c = colors[i % len(colors)]
        score_val = score.item() if torch.is_tensor(score) else score
        
        # Bounding Box
        x0, y0, x1, y1 = [int(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=c, width=3)
        draw.text((x0, max(0, y0-14)), f"{label} ({score_val:.2f})", fill=c)
        
        # Semi-transparente Maske
        color_layer = np.zeros((*mask.shape, 3), dtype=np.uint8)
        color_layer[...] = c
        alpha = (mask * 100).astype(np.uint8)
        img.paste(Image.fromarray(color_layer), mask=Image.fromarray(alpha))
    
    img.show(title="2D: Segmentierung")


def _generate_unique_colors(num_objects):
    """Generiert einzigartige Farben im HSV-Farbraum."""
    unique_colors = []
    for i in range(num_objects):
        hue = (i * 360.0 / max(num_objects, 1)) % 360
        saturation = 0.9
        value = 0.95
        
        h = hue / 60.0
        c = value * saturation
        x = c * (1 - abs((h % 2) - 1))
        m = value - c
        
        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        unique_colors.append([r + m, g + m, b + m])
    return unique_colors


def _load_pointcloud_data(session_path):
    """Lädt RGB, Depth und erstellt die 3D-Punktwolke."""
    rgb_path = os.path.join(session_path, "rgb", "rgb_0000.png")
    depth_path = os.path.join(session_path, "distance_to_image_plane", 
                               "distance_to_image_plane_0000.npy")
    
    rgb = np.array(Image.open(rgb_path))[:, :, :3]
    depth = np.load(depth_path)
    H, W = depth.shape
    
    # Kamera-Intrinsics
    fx = fy = 437.04
    cx, cy = W / 2, H / 2
    
    # Vollständige Punktwolke erstellen
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    all_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # Transformation (Open3D Konvention)
    all_points[:, 1] *= -1
    all_points[:, 2] *= -1
    
    return all_points, rgb, H, W


def visualize_3d_colored(session_path, masks, labels):
    """
    3D Visualisierung 1: Farbige Segmente (Oberflächen).
    """
    all_points, rgb, H, W = _load_pointcloud_data(session_path)
    
    # Basis-Punktwolke (grau)
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(all_points)
    gray = np.full((len(all_points), 3), 0.7)
    full_pcd.colors = o3d.utility.Vector3dVector(gray)
    
    geoms = [full_pcd]
    unique_colors = _generate_unique_colors(len(masks))
    assignment = np.full((H, W), -1, dtype=np.int32)
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask_np = np.asarray(mask)
        ys, xs = np.where((mask_np > 0) & (assignment == -1))
        
        if len(xs) == 0:
            continue
        
        assignment[ys, xs] = i
        linear_idx = ys * W + xs
        segment_points = all_points[linear_idx]
        
        pcd_segment = o3d.geometry.PointCloud()
        pcd_segment.points = o3d.utility.Vector3dVector(segment_points)
        color = unique_colors[i]
        
        if len(segment_points) > 100:
            pcd_surface = pcd_segment.voxel_down_sample(voxel_size=0.01)
            pcd_surface, _ = pcd_surface.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd_surface.colors = o3d.utility.Vector3dVector(
                np.tile(color, (len(pcd_surface.points), 1))
            )
        else:
            pcd_surface = pcd_segment
            pcd_surface.colors = o3d.utility.Vector3dVector(
                np.tile(color, (len(segment_points), 1))
            )
        
        geoms.append(pcd_surface)
    
    o3d.visualization.draw_geometries(geoms, window_name="1/3: Segmentierte Objekte (Farben)")


def visualize_3d_rgbd(session_path):
    """
    3D Visualisierung 2: RGBD Punktwolke mit Original-Bildfarben.
    """
    all_points, rgb, H, W = _load_pointcloud_data(session_path)
    
    # RGB-Farben normalisieren (0-255 -> 0-1)
    rgb_colors = rgb.reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
    
    o3d.visualization.draw_geometries([pcd], window_name="2/3: RGBD Punktwolke (Original-Farben)")


def visualize_3d_with_ids(session_path, masks, labels):
    """
    3D Visualisierung 3: Farbige Segmente mit ID-Nummern in der Mitte.
    """
    all_points, rgb, H, W = _load_pointcloud_data(session_path)
    
    # Basis-Punktwolke (grau)
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(all_points)
    gray = np.full((len(all_points), 3), 0.7)
    full_pcd.colors = o3d.utility.Vector3dVector(gray)
    
    geoms = [full_pcd]
    unique_colors = _generate_unique_colors(len(masks))
    assignment = np.full((H, W), -1, dtype=np.int32)
    mask_centers = []
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask_np = np.asarray(mask)
        ys, xs = np.where((mask_np > 0) & (assignment == -1))
        
        if len(xs) == 0:
            continue
        
        assignment[ys, xs] = i
        linear_idx = ys * W + xs
        segment_points = all_points[linear_idx]
        
        # Berechne Zentrum der Maske
        center_3d = segment_points.mean(axis=0)
        mask_centers.append((i + 1, center_3d, unique_colors[i]))  # ID beginnt bei 1
        
        pcd_segment = o3d.geometry.PointCloud()
        pcd_segment.points = o3d.utility.Vector3dVector(segment_points)
        color = unique_colors[i]
        
        if len(segment_points) > 100:
            pcd_surface = pcd_segment.voxel_down_sample(voxel_size=0.01)
            pcd_surface, _ = pcd_surface.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd_surface.colors = o3d.utility.Vector3dVector(
                np.tile(color, (len(pcd_surface.points), 1))
            )
        else:
            pcd_surface = pcd_segment
            pcd_surface.colors = o3d.utility.Vector3dVector(
                np.tile(color, (len(segment_points), 1))
            )
        
        geoms.append(pcd_surface)
    
    # Füge ID-Labels als flache weiße Scheiben hinzu
    for mask_id, center, color in mask_centers:
        # Erstelle flache weiße Scheibe (2D) als Marker
        disc = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.002)
        disc.translate(center)
        disc.paint_uniform_color([1.0, 1.0, 1.0])  # Weiße Scheibe
        disc.compute_vertex_normals()
        geoms.append(disc)
    
    # Zeige die Visualisierung mit Labels
    print("\n[VIZ] ID-Zuordnung:")
    for mask_id, center, color in mask_centers:
        print(f"  ID {mask_id}: Position ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    
    # Verwende Open3D Visualizer mit 3D-Labels
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3/3: Segmentierte Objekte mit IDs", width=1280, height=720)
    
    for geom in geoms:
        vis.add_geometry(geom)
    
    # Füge 3D-Labels hinzu (rote Schrift direkt auf der Scheibe)
    for mask_id, center, color in mask_centers:
        label_points = []
        label_colors = []
        red = [1.0, 0.0, 0.0]  # Rot statt neon-grün
        
        # Ziffern-Muster definieren (5x7 Grid)
        digit_patterns = {
            '1': [(2,0),(1,1),(2,1),(2,2),(2,3),(2,4),(2,5),(1,6),(2,6),(3,6)],
            '2': [(1,0),(2,0),(3,0),(0,1),(4,1),(4,2),(3,3),(2,4),(1,5),(0,6),(1,6),(2,6),(3,6),(4,6)],
            '3': [(0,0),(1,0),(2,0),(3,0),(4,1),(3,2),(2,3),(3,4),(4,5),(0,6),(1,6),(2,6),(3,6)],
            '4': [(0,0),(3,0),(0,1),(3,1),(0,2),(3,2),(0,3),(1,3),(2,3),(3,3),(4,3),(3,4),(3,5),(3,6)],
            '5': [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(0,2),(1,2),(2,2),(3,2),(4,3),(4,4),(0,5),(1,6),(2,6),(3,6)],
            '6': [(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(0,4),(4,4),(0,5),(4,5),(1,6),(2,6),(3,6)],
            '7': [(0,0),(1,0),(2,0),(3,0),(4,0),(4,1),(3,2),(2,3),(2,4),(2,5),(2,6)],
            '8': [(1,0),(2,0),(3,0),(0,1),(4,1),(0,2),(4,2),(1,3),(2,3),(3,3),(0,4),(4,4),(0,5),(4,5),(1,6),(2,6),(3,6)],
            '9': [(1,0),(2,0),(3,0),(0,1),(4,1),(0,2),(4,2),(1,3),(2,3),(3,3),(4,3),(4,4),(4,5),(1,6),(2,6),(3,6)],
            '0': [(1,0),(2,0),(3,0),(0,1),(4,1),(0,2),(4,2),(0,3),(4,3),(0,4),(4,4),(0,5),(4,5),(1,6),(2,6),(3,6)],
        }
        
        # Konvertiere ID zu String und erstelle Punkte
        id_str = str(mask_id)
        num_digits = len(id_str)
        scale = 0.006  # Etwas kleiner für die Scheibe
        total_width = num_digits * 5 * scale
        
        offset_x = 0
        for char in id_str:
            if char in digit_patterns:
                for (px, py) in digit_patterns[char]:
                    point = center.copy()
                    point[0] += (px + offset_x) * scale - total_width / 2  # Zentriert
                    point[1] -= (py - 3) * scale  # Vertikal zentriert
                    point[2] += 0.003  # Direkt auf der Scheibe
                    label_points.append(point)
                    label_colors.append(red)
            offset_x += 6  # Abstand zwischen Ziffern
        
        if label_points:
            label_pcd = o3d.geometry.PointCloud()
            label_pcd.points = o3d.utility.Vector3dVector(np.array(label_points))
            label_pcd.colors = o3d.utility.Vector3dVector(np.array(label_colors))
            vis.add_geometry(label_pcd)
    
    vis.run()
    vis.destroy_window()


def visualize_3d(session_path, masks, labels):
    """
    Hauptfunktion: Zeigt alle 3 Visualisierungen nacheinander.
    """
    print("\n[VIZ] Starte 3-fache Visualisierung...")
    
    # Visualisierung 1: Farbige Segmente
    visualize_3d_colored(session_path, masks, labels)
    
    # Visualisierung 2: RGBD mit Original-Farben
    visualize_3d_rgbd(session_path)
    
    # Visualisierung 3: Farbige Segmente mit IDs
    visualize_3d_with_ids(session_path, masks, labels)
    
    print("[VIZ] Alle 3 Visualisierungen abgeschlossen.\n")
