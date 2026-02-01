"""
Visualization/visualizer.py
Modul für 2D und 3D Visualisierung von Segmentierungsergebnissen.
"""
import numpy as np
import torch
import open3d as o3d
import os
import json
from datetime import datetime
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


def visualize_3d_colored(session_path, masks, labels, window_name="Segmentierte Objekte"):
    """
    3D Visualisierung: Farbige Segmente (Oberflächen).
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
    
    o3d.visualization.draw_geometries(geoms, window_name=window_name)


def visualize_3d_rgbd(session_path):
    """
    3D Visualisierung 1: RGBD Punktwolke mit Original-Bildfarben.
    """
    all_points, rgb, H, W = _load_pointcloud_data(session_path)
    
    # RGB-Farben normalisieren (0-255 -> 0-1)
    rgb_colors = rgb.reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
    
    o3d.visualization.draw_geometries([pcd], window_name="1/6: RGBD Punktwolke (Original-Farben)")


def visualize_dino_boxes(session_path, dino_debug, stage="raw"):
    """
    3D Visualisierung: Grounding DINO Bounding Boxes als RAHMEN (nicht gefüllt).
    
    Args:
        session_path: Pfad zur Session
        dino_debug: Debug-Daten von run_grounding_dino_only
        stage: "raw" für Raw-Boxes, "post_size" für nach Größenfilter, "post_iou" für nach NMS
    """
    if dino_debug is None:
        print(f"[VIZ] Keine DINO Debug-Daten vorhanden für Stage: {stage}")
        return
    
    all_points, rgb, H, W = _load_pointcloud_data(session_path)
    
    # Lade Tiefenbild für Z-Koordinaten der Box-Rahmen
    depth_path = os.path.join(session_path, "distance_to_image_plane", 
                               "distance_to_image_plane_0000.npy")
    depth = np.load(depth_path)
    
    # Kamera-Intrinsics (identisch zu _load_pointcloud_data)
    fx = fy = 437.04
    cx, cy = W / 2, H / 2
    
    # Wähle die richtige Box-Liste
    if stage == "raw":
        boxes = dino_debug.get("raw_boxes", [])
        labels = dino_debug.get("raw_labels", [])
        window_title = "2/6: DINO Raw Boxes (vor Filterung)"
    elif stage == "post_size":
        boxes = dino_debug.get("post_size_filter_boxes", [])
        labels = dino_debug.get("post_size_filter_labels", [])
        window_title = "3/6: DINO Boxes (nach Größen-Filter)"
    else:  # post_iou
        boxes = dino_debug.get("post_iou_boxes", [])
        labels = dino_debug.get("post_iou_labels", [])
        window_title = "3/6: DINO Boxes (nach IoU-NMS)"
    
    if not boxes:
        print(f"[VIZ] Keine Boxes für Stage: {stage}")
        return
    
    # Basis-Punktwolke mit RGBD Farben (nicht grau!)
    rgb_colors = rgb.reshape(-1, 3) / 255.0
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(all_points)
    full_pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
    
    geoms = [full_pcd]
    unique_colors = _generate_unique_colors(len(boxes))
    
    # Jede Box als 3D-Rahmen (Linien) anzeigen
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(c) for c in box]
        
        # Sichere Grenzen
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        
        # Finde die durchschnittliche Tiefe entlang der Box-Kanten
        # Damit der Rahmen auf der richtigen 3D-Höhe gezeichnet wird
        edge_depths = []
        
        # Obere Kante
        for px in range(x1, x2+1, 5):
            if depth[y1, px] > 0:
                edge_depths.append(depth[y1, px])
        # Untere Kante
        for px in range(x1, x2+1, 5):
            if depth[y2, px] > 0:
                edge_depths.append(depth[y2, px])
        # Linke Kante
        for py in range(y1, y2+1, 5):
            if depth[py, x1] > 0:
                edge_depths.append(depth[py, x1])
        # Rechte Kante
        for py in range(y1, y2+1, 5):
            if depth[py, x2] > 0:
                edge_depths.append(depth[py, x2])
        
        if not edge_depths:
            continue
        
        # Nimm minimale Tiefe (oberste Oberfläche) für den Rahmen
        box_z = np.percentile(edge_depths, 10)
        
        # Konvertiere 2D Box-Ecken zu 3D Punkten
        def pixel_to_3d(px, py, z):
            x_3d = (px - cx) * z / fx
            y_3d = (py - cy) * z / fy
            # Open3D Transformation
            return [x_3d, -y_3d, -z]
        
        # 4 Ecken der Box
        corners = [
            pixel_to_3d(x1, y1, box_z),  # Top-left
            pixel_to_3d(x2, y1, box_z),  # Top-right
            pixel_to_3d(x2, y2, box_z),  # Bottom-right
            pixel_to_3d(x1, y2, box_z),  # Bottom-left
        ]
        
        # Erstelle LineSet für den Rahmen
        lines = [
            [0, 1],  # Top
            [1, 2],  # Right
            [2, 3],  # Bottom
            [3, 0],  # Left
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Farbe für die Linien
        color = unique_colors[i]
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
        
        geoms.append(line_set)
        
        # Optional: ID-Label als Text-Punkt in der Mitte der Box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center_3d = pixel_to_3d(center_x, center_y, box_z - 0.02)  # Etwas vor der Box
        
        # Kleine Punktwolke für das Label (ID-Nummer)
        label_pcd = o3d.geometry.PointCloud()
        label_pcd.points = o3d.utility.Vector3dVector([center_3d])
        label_pcd.colors = o3d.utility.Vector3dVector([color])
        geoms.append(label_pcd)
    
    print(f"[VIZ] Zeige {len(boxes)} DINO Box-Rahmen ({stage})")
    o3d.visualization.draw_geometries(geoms, window_name=window_title)


def visualize_sobel_edges(session_path, viz_data, window_title_suffix=""):
    """
    3D Visualisierung: Gradienten/Kanten Analyse.
    Färbt die Punktwolke basierend auf der Gradienten-Magnitude.
    """
    if viz_data is None:
        print("[VIZ] Keine Sobel-Daten vorhanden.")
        return

    all_points, rgb, H, W = _load_pointcloud_data(session_path)
    
    gradient = viz_data["gradient_magnitude"]
    edges = viz_data["edges"]
    
    # Gradienten normalisieren für Farb-Mapping (0-1)
    # Clip bei 50mm für Kontrast
    grad_norm = np.clip(gradient, 0, 50) / 50.0
    
    # Colormap: Blau (flach) -> Rot (Kante)
    colors = np.zeros((H * W, 3))
    
    grad_flat = grad_norm.flatten()
    colors[:, 0] = grad_flat       # Rot
    colors[:, 2] = 1 - grad_flat   # Blau
    
    # Markiere erkannte Edges (Spalten) in hellem Grün
    edges_flat = edges.flatten()
    colors[edges_flat > 0, 0] = 0
    colors[edges_flat > 0, 1] = 1
    colors[edges_flat > 0, 2] = 0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    title = f"5/6: Gradienten/Spalten Analyse{window_title_suffix} (Blau=Flach, Rot=Steil, Grün=Kante)"
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_per_box_gradient(session_path, viz_data, dino_debug):
    """
    NEU: Zeigt die Gradienten-Analyse pro Box an.
    Hebt hervor, wo starke Tiefensprünge innerhalb von Boxes gefunden wurden.
    """
    if viz_data is None or "per_box_analysis" not in viz_data:
        print("[VIZ] Keine Per-Box Gradient-Daten vorhanden.")
        return
    
    per_box = viz_data.get("per_box_analysis", [])
    if not per_box:
        print("[VIZ] Keine Per-Box Analyse durchgeführt.")
        return
    
    all_points, rgb, H, W = _load_pointcloud_data(session_path)
    
    # Basis: Graue Punktwolke
    base_colors = np.full((H, W, 3), 0.5)  # Grau
    
    # Für jede Box: Zeige gefundene Segmente
    unique_colors = _generate_unique_colors(len(per_box) * 3)  # Genug Farben für alle Segmente
    color_idx = 0
    
    for box_idx, analysis in enumerate(per_box):
        if analysis is None:
            continue
            
        x1, y1, x2, y2 = analysis['box_coords']
        segment_labels = analysis['segment_labels']
        split_mask = analysis['split_mask']
        
        # Färbe die Split-Linien weiß
        box_h, box_w = split_mask.shape
        for by in range(box_h):
            for bx in range(box_w):
                img_y = y1 + by
                img_x = x1 + bx
                if img_y < H and img_x < W:
                    if split_mask[by, bx] > 0:
                        base_colors[img_y, img_x] = [1.0, 1.0, 1.0]  # Weiß = Split-Linie
                    else:
                        seg_id = segment_labels[by, bx]
                        if seg_id > 0:
                            # Jedes Segment bekommt eigene Farbe
                            c_idx = (box_idx * 3 + seg_id) % len(unique_colors)
                            base_colors[img_y, img_x] = unique_colors[c_idx]
    
    # Zu 1D umformen
    colors_flat = base_colors.reshape(-1, 3)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(colors_flat)
    
    segments_found = sum(a['num_segments'] for a in per_box if a)
    title = f"4/6: Per-Box Gradient-Analyse ({segments_found} Segmente total, Weiß=Trennlinien)"
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_3d(session_path, refined_masks, refined_labels, sobel_viz_data=None, 
                 original_masks=None, original_labels=None, dino_debug=None):
    """
    Hauptfunktion: Zeigt alle 6 Visualisierungen nacheinander.
    
    Reihenfolge:
    1. RGBD mit Original-Farben
    2. DINO Raw Boxes (alle erkannten)
    3. DINO Boxes nach IoU-NMS
    4. Per-Box Gradient-Analyse (wo wurden Segmente getrennt?)
    5. Globale Gradient/Edges Analyse
    6. Finale segmentierte Objekte (nach Refinement)
    """
    print("\n[VIZ] Starte 6-fache Debug-Visualisierung...")
    
    # 1. RGBD mit Original-Farben
    print("[VIZ] 1/6: RGBD Punktwolke...")
    visualize_3d_rgbd(session_path)
    
    # 2. DINO Raw Boxes
    if dino_debug:
        print("[VIZ] 2/6: DINO Raw Boxes...")
        visualize_dino_boxes(session_path, dino_debug, stage="raw")
    
    # 3. DINO Boxes nach IoU-NMS
    if dino_debug:
        print("[VIZ] 3/6: DINO Boxes nach IoU-NMS...")
        visualize_dino_boxes(session_path, dino_debug, stage="post_iou")
    
    # 4. Per-Box Gradient-Analyse
    if sobel_viz_data and "per_box_analysis" in sobel_viz_data:
        print("[VIZ] 4/6: Per-Box Gradient-Analyse...")
        visualize_per_box_gradient(session_path, sobel_viz_data, dino_debug)
    
    # 5. Globale Gradient/Edges Analyse
    if sobel_viz_data:
        print("[VIZ] 5/6: Globale Gradient-Analyse...")
        visualize_sobel_edges(session_path, sobel_viz_data)
        
    # 6. Finale segmentierte Objekte (mit Sobel verfeinert)
    print("[VIZ] 6/6: Finale Segmente...")
    visualize_3d_colored(session_path, refined_masks, refined_labels, 
                        window_name="6/6: Finale Segmente (Nach Gradient-Refinement)")
    
    print("[VIZ] Alle 6 Visualisierungen abgeschlossen.\n")
    return {}


def _load_viewpoints():
    """Lädt die kalibrierten Viewpoints aus JSON."""
    viewpoints_path = os.path.join(os.path.dirname(__file__), "viewpoints.json")
    
    if not os.path.exists(viewpoints_path):
        raise FileNotFoundError(f"Viewpoints nicht gefunden: {viewpoints_path}\n"
                               "Bitte zuerst calibrate_viewpoints.py ausführen.")
    
    with open(viewpoints_path, "r") as f:
        data = json.load(f)
    
    viewpoints = {}
    for key in ["1", "2", "3"]:
        if key in data:
            viewpoints[key] = {
                "extrinsic": np.array(data[key]["extrinsic"]),
                "intrinsic": np.array(data[key]["intrinsic"]),
                "width": data[key]["width"],
                "height": data[key]["height"]
            }
    
    return viewpoints


def capture_scene_screenshots(session_path, masks, labels, output_dir=None):
    """
    Erstellt 3 Screenshots der gesamten Szene mit allen Objekten und IDs 
    aus den kalibrierten Viewpoints.
    
    Args:
        session_path: Pfad zur Session
        masks: Liste von Masken
        labels: Liste von Labels
        output_dir: Ausgabeordner (optional, default: session_path/screenshots)
    
    Returns:
        Liste mit Pfaden zu den 3 Screenshots
    """
    print("\n[SCREENSHOT] Starte Screenshot-Aufnahme der Szene...")
    
    # Viewpoints laden
    viewpoints = _load_viewpoints()
    if len(viewpoints) < 3:
        print("[WARNUNG] Weniger als 3 Viewpoints definiert!")
    
    # Output-Verzeichnis
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(session_path, "screenshots", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Punktwolke laden
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
    
    # Ziffern-Muster (identisch zu visualize_3d_with_ids)
    digit_patterns = {
        '1': [(1,0),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(1,6),(2,6),(3,6),(2,0),(2,1)],
        '2': [(0,0),(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(3,3),(2,3),(1,4),(0,5),(0,6),(1,6),(2,6),(3,6),(4,6)],
        '3': [(0,0),(1,0),(2,0),(3,0),(4,1),(4,2),(2,3),(3,3),(4,4),(4,5),(0,6),(1,6),(2,6),(3,6)],
        '4': [(0,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(4,3),(3,0),(3,1),(3,2),(3,4),(3,5),(3,6)],
        '5': [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(4,4),(4,5),(0,6),(1,6),(2,6),(3,6)],
        '6': [(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(0,4),(4,4),(0,5),(4,5),(1,6),(2,6),(3,6)],
        '7': [(0,0),(1,0),(2,0),(3,0),(4,0),(4,1),(3,2),(3,3),(2,4),(2,5),(2,6)],
        '8': [(1,0),(2,0),(3,0),(0,1),(4,1),(0,2),(4,2),(1,3),(2,3),(3,3),(0,4),(4,4),(0,5),(4,5),(1,6),(2,6),(3,6)],
        '9': [(1,0),(2,0),(3,0),(0,1),(4,1),(0,2),(4,2),(1,3),(2,3),(3,3),(4,3),(4,4),(4,5),(1,6),(2,6),(3,6)],
        '0': [(1,0),(2,0),(3,0),(0,1),(4,1),(0,2),(4,2),(0,3),(4,3),(0,4),(4,4),(0,5),(4,5),(1,6),(2,6),(3,6)],
    }
    
    # Objekte mit Farben und IDs erstellen
    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask_np = np.asarray(mask)
        ys, xs = np.where((mask_np > 0) & (assignment == -1))
        
        if len(xs) == 0:
            continue
        
        assignment[ys, xs] = i
        linear_idx = ys * W + xs
        segment_points = all_points[linear_idx]
        
        center_3d = segment_points.mean(axis=0)
        mask_centers.append((i + 1, center_3d, unique_colors[i]))
        
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
    
    # ID-Scheiben und Labels erstellen
    for mask_id, center, _ in mask_centers:
        disc = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.002)
        disc_offset = center.copy()
        disc_offset[2] += 0.05
        disc.translate(disc_offset)
        disc.paint_uniform_color([1.0, 1.0, 1.0])
        disc.compute_vertex_normals()
        geoms.append(disc)
        
        # Label-Punkte
        label_points = []
        label_colors = []
        red = [1.0, 0.0, 0.0]
        
        id_str = str(mask_id)
        num_digits = len(id_str)
        scale = 0.008
        total_width = num_digits * 5 * scale
        
        offset_x = 0
        for char in id_str:
            if char in digit_patterns:
                for (px, py) in digit_patterns[char]:
                    rel_x = -((px + offset_x) * scale - total_width / 2)
                    rel_y = (py - 3) * scale
                    for dx in [-0.0005, 0, 0.0005]:
                        for dy in [-0.0005, 0, 0.0005]:
                            label_points.append([disc_offset[0] + rel_x + dx, 
                                                disc_offset[1] + rel_y + dy, 
                                                disc_offset[2] + 0.005])
                            label_colors.append(red)
            offset_x += 6
        
        if label_points:
            label_pcd = o3d.geometry.PointCloud()
            label_pcd.points = o3d.utility.Vector3dVector(np.array(label_points))
            label_pcd.colors = o3d.utility.Vector3dVector(np.array(label_colors))
            geoms.append(label_pcd)
    
    # Visualizer erstellen
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    
    for geom in geoms:
        vis.add_geometry(geom)
    
    # Render-Optionen
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    screenshot_paths = []
    
    # Screenshots aus allen 3 Viewpoints
    for vp_key, vp_data in viewpoints.items():
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        cam.extrinsic = vp_data["extrinsic"]
        ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
        
        vis.poll_events()
        vis.update_renderer()
        
        screenshot_name = f"scene_viewpoint_{vp_key}.png"
        screenshot_path = os.path.join(output_dir, screenshot_name)
        vis.capture_screen_image(screenshot_path, do_render=True)
        screenshot_paths.append(screenshot_path)
        print(f"  [SCREENSHOT] Viewpoint {vp_key} → {screenshot_name}")
    
    vis.destroy_window()
    
    print(f"[SCREENSHOT] Fertig! 3 Screenshots gespeichert in: {output_dir}\n")
    
    return screenshot_paths
