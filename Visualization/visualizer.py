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


def visualize_3d(session_path, masks, labels):
    """
    3D Visualisierung mit farbigen Segmenten.
    
    Args:
        session_path: Pfad zur Session (enthält rgb/ und distance_to_image_plane/)
        masks: Liste von binären Masken
        labels: Liste von Label-Strings
    """
    # Lade Daten
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
    
    # Basis-Punktwolke (grau)
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(all_points)
    gray = np.full((len(all_points), 3), 0.7)
    full_pcd.colors = o3d.utility.Vector3dVector(gray)
    
    geoms = [full_pcd]
    
    # Farben für Segmente
    distinct_colors = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.5, 0.0], [0.5, 0.0, 1.0]
    ]
    
    # Erstelle Zuordnungsmaske (verhindert Überlappung)
    assignment = np.full((H, W), -1, dtype=np.int32)
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask_np = np.asarray(mask)
        
        # Nur Pixel verwenden, die noch keinem Objekt zugeordnet sind
        ys, xs = np.where((mask_np > 0) & (assignment == -1))
        
        if len(xs) == 0:
            continue
        
        # Markiere als zugeordnet
        assignment[ys, xs] = i
        
        # Punkte extrahieren
        linear_idx = ys * W + xs
        segment_points = all_points[linear_idx]
        
        # Erstelle farbige Punktwolke
        pcd_segment = o3d.geometry.PointCloud()
        pcd_segment.points = o3d.utility.Vector3dVector(segment_points)
        
        color = distinct_colors[i % len(distinct_colors)]
        pcd_segment.colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(segment_points), 1))
        )
        
        geoms.append(pcd_segment)
    
    o3d.visualization.draw_geometries(geoms, window_name="3D: Segmentierte Objekte")
