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
    
    # Zeige die Visualisierung mit Labels
    print("\n[VIZ] ID-Zuordnung:")
    for mask_id, center, color in mask_centers:
        print(f"  ID {mask_id}: Position ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    
    # Ziffern-Muster definieren (5x7 Grid) - DICHTERE Muster
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
    
    # Erstelle Billboard-Labels (Scheibe + Text) die zur Kamera ausgerichtet werden
    billboard_data = []  # (center, disc_mesh, label_pcd, base_disc_vertices, base_label_points)
    
    for mask_id, center, _ in mask_centers:
        # Erstelle Scheibe - Position so dass sie aus der Oberfläche heraussteht
        disc = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.002)
        # Offset: Scheibe steht aus der Oberfläche heraus (in Kamera-Richtung / +Z)
        disc_offset = center.copy()
        disc_offset[2] += 0.05  # 5cm über der Oberfläche
        disc.translate(disc_offset)
        disc.paint_uniform_color([1.0, 1.0, 1.0])
        disc.compute_vertex_normals()
        base_disc_vertices = np.asarray(disc.vertices).copy()
        
        # Erstelle Label-Punkte - DICHTER mit mehr Punkten pro Position
        label_points = []
        label_colors = []
        red = [1.0, 0.0, 0.0]
        
        id_str = str(mask_id)
        num_digits = len(id_str)
        scale = 0.008  # Größere Skalierung
        total_width = num_digits * 5 * scale
        
        offset_x = 0
        for char in id_str:
            if char in digit_patterns:
                for (px, py) in digit_patterns[char]:
                    # Relative Position zur Mitte
                    # X: NEGIERT für korrekte Lesbarkeit (von vorne gesehen)
                    rel_x = -((px + offset_x) * scale - total_width / 2)
                    rel_y = (py - 3) * scale
                    # Mehrere Punkte pro Position für dichtere Darstellung (feinere Auflösung)
                    for dx in [-0.0005, 0, 0.0005]:
                        for dy in [-0.0005, 0, 0.0005]:
                            # Vorderseite
                            label_points.append([rel_x + dx, rel_y + dy, 0.005])
                            label_colors.append(red)
                            # Rückseite (X nochmal gespiegelt)
                            label_points.append([-rel_x - dx, rel_y + dy, -0.005])
                            label_colors.append(red)
            offset_x += 6
        
        label_pcd = o3d.geometry.PointCloud()
        label_colors_arr = np.array(label_colors) if label_colors else np.array([[1, 0, 0]])
        if label_points:
            base_label_points = np.array(label_points)
            label_pcd.points = o3d.utility.Vector3dVector(base_label_points + disc_offset)
            label_pcd.colors = o3d.utility.Vector3dVector(label_colors_arr)
        else:
            base_label_points = np.array([[0, 0, 0]])
        
        billboard_data.append((disc_offset, disc, label_pcd, base_disc_vertices, base_label_points, label_colors_arr))
        geoms.append(disc)
    
    # Verwende Open3D Visualizer mit Animation Callback für Billboard-Effekt
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3/3: Segmentierte Objekte mit IDs", width=1280, height=720)
    
    for geom in geoms:
        vis.add_geometry(geom)
    
    for _, _, label_pcd, _, base_label_pts, _ in billboard_data:
        if len(base_label_pts) > 1 or (len(base_label_pts) == 1 and not np.allclose(base_label_pts[0], [0, 0, 0])):
            vis.add_geometry(label_pcd)
    
    def update_billboards(vis):
        """Callback um Billboards zur Kamera auszurichten."""
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = cam.extrinsic
        
        # Kamera-Position und Blickrichtung extrahieren
        R = extrinsic[:3, :3]
        # Rotation Matrix um Billboards zur Kamera auszurichten
        
        for center, disc, label_pcd, base_disc_verts, base_label_pts, label_colors_arr in billboard_data:
            # Rotiere Scheibe zur Kamera
            rotated_verts = (R.T @ (base_disc_verts - center).T).T + center
            disc.vertices = o3d.utility.Vector3dVector(rotated_verts)
            disc.compute_vertex_normals()
            vis.update_geometry(disc)
            
            # Rotiere Label-Punkte und stelle Farben wieder her
            if len(base_label_pts) > 1 or (len(base_label_pts) == 1 and not np.allclose(base_label_pts[0], [0, 0, 0])):
                rotated_pts = (R.T @ base_label_pts.T).T + center
                label_pcd.points = o3d.utility.Vector3dVector(rotated_pts)
                label_pcd.colors = o3d.utility.Vector3dVector(label_colors_arr)
                vis.update_geometry(label_pcd)
        
        return False
    
    vis.register_animation_callback(update_billboards)
    vis.run()
    vis.destroy_window()
    
    # Erstelle Resultat-Dictionary
    results = {}
    for mask_id, center, color in mask_centers:
        # ID ist 1-basiert, mask index ist 0-basiert
        idx = mask_id - 1
        results[mask_id] = {
            "mask": masks[idx],
            "label": labels[idx] if idx < len(labels) else "unknown",
            "center_3d": center.tolist(),
            "color": color
        }
    
    return results


def visualize_3d(session_path, masks, labels):
    """
    Hauptfunktion: Zeigt alle 3 Visualisierungen nacheinander.
    Git ein Dictionary mit ID-Zuordnungen zurück.
    """
    print("\n[VIZ] Starte 3-fache Visualisierung...")
    
    # Visualisierung 1: Farbige Segmente
    visualize_3d_colored(session_path, masks, labels)
    
    # Visualisierung 2: RGBD mit Original-Farben
    visualize_3d_rgbd(session_path)
    
    # Visualisierung 3: Farbige Segmente mit IDs
    results = visualize_3d_with_ids(session_path, masks, labels)
    
    print("[VIZ] Alle 3 Visualisierungen abgeschlossen.\n")
    return results
