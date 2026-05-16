"""
Visualization/visualizer.py
Modul für 2D und 3D Visualisierung von Segmentierungsergebnissen.
"""
import numpy as np
import torch
import open3d as o3d
import cv2
import os
import json
from datetime import datetime
from PIL import Image, ImageDraw
from Segmentation.sobel_refinement import (
    analyze_box_gradient_z_aligned,
    align_mask_to_depth_plane,
    select_frontmost_segment,
)
from config import (
    Z_ALIGN_MIN_KEEP_RATIO,
    MATCH_CLOSURE_RATIO,
    MATCH_BORDER_TOUCH_RATIO,
    MATCH_DEDUP_IOU,
    MATCH_DEDUP_CONTAINMENT,
    MATCH_DEDUP_IOU_FAR,
    MATCH_DEDUP_CONTAINMENT_FAR,
    MATCH_DEDUP_IOU_DEEP,
    MATCH_DEDUP_CONTAINMENT_DEEP,
    MATCH_DEDUP_Z_DEEP_M,
    MATCH_DEDUP_Z_OCCLUDE_M,
    MATCH_DEDUP_Z_DIFF_M,
    MATCH_DEDUP_USE_DINO_BBOX,
    MATCH_DEDUP_USE_BBOX,
    MATCH_DEDUP_KEEP_CLOSER,
)


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


def _dim_colors_outside_workspace(colors, workspace_mask):
    """Außerhalb des Workspace: Punkte abdunkeln."""
    if workspace_mask is None:
        return colors
    outside = ~workspace_mask.reshape(-1)
    colors = colors.copy()
    colors[outside] *= 0.25
    return colors


def _load_pointcloud_data(session_path, session_context=None):
    """Lädt RGB, Depth und erstellt die 3D-Punktwolke."""
    rgb_path = os.path.join(session_path, "rgb", "rgb_0000.png")
    rgb = np.array(Image.open(rgb_path))[:, :, :3]
    if session_context is not None:
        depth = session_context.depth_abs
    else:
        depth_path = os.path.join(
            session_path,
            "distance_to_image_plane",
            "distance_to_image_plane_0000.npy",
        )
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
    
    workspace_mask = session_context.workspace_mask if session_context is not None else None
    return all_points, rgb, H, W, workspace_mask


def _load_pcd_for_viz(session_path, session_context=None):
    """Punktwolke + Basisfarben (Workspace-Ränder abgedunkelt)."""
    all_points, rgb, H, W, workspace_mask = _load_pointcloud_data(
        session_path, session_context
    )
    base_colors = _dim_colors_outside_workspace(rgb.reshape(-1, 3) / 255.0, workspace_mask)
    return all_points, rgb, H, W, base_colors, workspace_mask


def _camera_to_o3d(points: np.ndarray) -> np.ndarray:
    """Kamera-Koordinaten → Open3D-Anzeige (Y/Z invertiert, wie übrige Viz)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    out = pts.copy()
    out[:, 1] *= -1
    out[:, 2] *= -1
    return out


_BOTTOM_METHOD_COLORS = {
    "measured": [0.25, 0.95, 0.35],
    "from_neighbor": [0.25, 0.55, 1.0],
    "from_pallet": [1.0, 0.85, 0.2],
    "uncertain": [1.0, 0.3, 0.3],
}


_OBB_EDGE_INDICES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

_OBB_FACE_TRIANGLES = (
    (0, 1, 2), (0, 2, 3),
    (4, 6, 5), (4, 7, 6),
    (0, 4, 5), (0, 5, 1),
    (1, 5, 6), (1, 6, 2),
    (2, 6, 7), (2, 7, 3),
    (3, 7, 4), (3, 4, 0),
)


def _make_obb_wireframe(corners_cam: np.ndarray, color: list[float]) -> o3d.geometry.LineSet:
    corners = _camera_to_o3d(corners_cam)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(list(_OBB_EDGE_INDICES))
    ls.colors = o3d.utility.Vector3dVector([color for _ in _OBB_EDGE_INDICES])
    return ls


def _make_obb_solid_mesh(
    corners_cam: np.ndarray,
    color: list[float],
    shade: float = 0.45,
) -> o3d.geometry.TriangleMesh:
    """Vollflächiger Quader-Mesh (alle 6 Flächen) mit abgedunkelter Farbe."""
    corners = _camera_to_o3d(corners_cam)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(list(_OBB_FACE_TRIANGLES))
    mesh.paint_uniform_color([c * shade for c in color])
    mesh.compute_vertex_normals()
    return mesh


def _make_cylinder_edge(p0: np.ndarray, p1: np.ndarray, radius: float, color: list[float]):
    """Erzeugt ein Zylinder-Mesh zwischen p0 und p1 (für dicke Kanten)."""
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=12, split=1)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(color)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = vec / length
    rot_axis = np.cross(z_axis, axis)
    rot_axis_norm = np.linalg.norm(rot_axis)
    if rot_axis_norm < 1e-9:
        if axis[2] < 0:
            R = np.diag([1.0, -1.0, -1.0])
            cyl.rotate(R, center=np.zeros(3))
    else:
        rot_axis /= rot_axis_norm
        angle = float(np.arccos(np.clip(np.dot(z_axis, axis), -1.0, 1.0)))
        K = np.array([
            [0.0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0.0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0.0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        cyl.rotate(R, center=np.zeros(3))
    midpoint = 0.5 * (p0 + p1)
    cyl.translate(midpoint)
    return cyl


def _make_obb_thick_edges(
    corners_cam: np.ndarray,
    color: list[float],
    radius: float = 0.004,
) -> list:
    """12 Zylinder-Meshes für gut sichtbare Box-Kanten."""
    corners = _camera_to_o3d(corners_cam)
    cylinders = []
    for i, j in _OBB_EDGE_INDICES:
        cyl = _make_cylinder_edge(corners[i], corners[j], radius, color)
        if cyl is not None:
            cylinders.append(cyl)
    return cylinders


def _make_obb_corner_markers(
    corners_cam: np.ndarray,
    color: list[float],
    radius: float = 0.008,
) -> list:
    corners = _camera_to_o3d(corners_cam)
    spheres = []
    for p in corners:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
        s.compute_vertex_normals()
        s.paint_uniform_color(color)
        s.translate(p)
        spheres.append(s)
    return spheres


def visualize_bottom_inference_3d(
    session_path,
    candidates,
    window_name: str = "Bottom-Plane Inference",
    session_context=None,
):
    """
    3D-Ansicht: extrudierte Paket-OBBs nach Stage 2.5 (Top-Fläche + inferierte Bodenhöhe).
    """
    if not candidates:
        print("[VIZ] Bottom-Inference: keine Kandidaten – übersprungen.")
        return

    all_points, _, _, _, base_colors, _ = _load_pcd_for_viz(session_path, session_context)
    bg = o3d.geometry.PointCloud()
    bg.points = o3d.utility.Vector3dVector(all_points)
    bg.colors = o3d.utility.Vector3dVector(base_colors * 0.45)
    geoms: list = [bg]

    if session_context is not None:
        z_pal = float(session_context.z_pallet_m)
        pal_pts = _camera_to_o3d(
            np.array([[-0.4, 0.0, z_pal], [0.4, 0.0, z_pal], [0.0, -0.3, z_pal], [0.0, 0.3, z_pal]])
        )
        pal_ls = o3d.geometry.LineSet()
        pal_ls.points = o3d.utility.Vector3dVector(pal_pts)
        pal_ls.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3]])
        pal_ls.paint_uniform_color([0.6, 0.6, 0.6])
        geoms.append(pal_ls)

    method_counts: dict[str, int] = {}
    for i, cand in enumerate(candidates):
        if cand.bottom is None:
            continue
        b = cand.bottom
        method = b.bottom_method
        method_counts[method] = method_counts.get(method, 0) + 1
        color = _BOTTOM_METHOD_COLORS.get(method, [0.8, 0.8, 0.8])

        pts = np.asarray(cand.points_3d, dtype=np.float64)
        if len(pts) > 0:
            pcd_top = o3d.geometry.PointCloud()
            pcd_top.points = o3d.utility.Vector3dVector(_camera_to_o3d(pts))
            if len(pts) > 200:
                pcd_top = pcd_top.voxel_down_sample(voxel_size=0.008)
            pcd_top.paint_uniform_color([min(1.0, c + 0.1) for c in color])
            geoms.append(pcd_top)

        corners = np.asarray(b.parcel_obb["corners_3d"], dtype=np.float64)
        geoms.append(_make_obb_solid_mesh(corners, color, shade=0.35))
        geoms.extend(_make_obb_thick_edges(corners, color, radius=0.005))
        geoms.extend(_make_obb_corner_markers(corners, color, radius=0.009))

        label = cand.debug.get("label", cand.candidate_id[:6])
        case = cand.debug.get("case_label", "?")
        print(
            f"  [BOTTOM-VIZ] #{i} {label}: {method} conf={b.bottom_confidence:.2f} "
            f"h={b.height_m:.3f}m case={case}"
        )

    dist = ", ".join(f"{k}={v}" for k, v in sorted(method_counts.items()))
    title = f"{window_name} ({dist})" if dist else window_name
    print(f"[VIZ] Bottom-Inference: {len(candidates)} Kandidaten, {dist}")
    o3d.visualization.draw_geometries(geoms, window_name=title)


def visualize_3d_colored(
    session_path, masks, labels, window_name="Segmentierte Objekte", session_context=None
):
    """
    3D Visualisierung: Farbige Segmente (Oberflächen).
    """
    all_points, _, H, W, base_colors, _ = _load_pcd_for_viz(session_path, session_context)

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(all_points)
    full_pcd.colors = o3d.utility.Vector3dVector(base_colors)
    
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


def visualize_3d_rgbd(session_path, session_context=None):
    """
    3D Visualisierung 1: RGBD Punktwolke mit Original-Bildfarben.
    """
    all_points, _, _, _, base_colors, _ = _load_pcd_for_viz(session_path, session_context)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(base_colors)
    
    o3d.visualization.draw_geometries([pcd], window_name="1/6: RGBD Punktwolke (Original-Farben)")


def visualize_dino_boxes(session_path, dino_debug, stage="raw", session_context=None):
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
    
    all_points, rgb, H, W, base_colors, _ = _load_pcd_for_viz(session_path, session_context)
    if session_context is not None:
        depth = session_context.depth_abs
    else:
        depth = np.load(
            os.path.join(
                session_path,
                "distance_to_image_plane",
                "distance_to_image_plane_0000.npy",
            )
        )
    
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
    
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(all_points)
    full_pcd.colors = o3d.utility.Vector3dVector(base_colors)
    
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


def _build_dino_box_wireframes(boxes, depth, H, W, box_colors=None, z_planes=None):
    """Erstellt 3D-LineSets für Bounding Boxes (optional feste Z-Ebene pro Box)."""
    fx = fy = 437.04
    cx, cy = W / 2, H / 2

    if box_colors is None:
        box_colors = _generate_unique_colors(len(boxes))

    def pixel_to_3d(px, py, z):
        x_3d = (px - cx) * z / fx
        y_3d = (py - cy) * z / fy
        return [x_3d, -y_3d, -z]

    geoms = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(c) for c in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        if z_planes is not None and i < len(z_planes) and z_planes[i] is not None:
            box_z = z_planes[i]
        else:
            edge_depths = []
            for px in range(x1, x2 + 1, 5):
                if depth[y1, px] > 0:
                    edge_depths.append(depth[y1, px])
                if depth[y2, px] > 0:
                    edge_depths.append(depth[y2, px])
            for py in range(y1, y2 + 1, 5):
                if depth[py, x1] > 0:
                    edge_depths.append(depth[py, x1])
                if depth[py, x2] > 0:
                    edge_depths.append(depth[py, x2])
            if not edge_depths:
                continue
            box_z = np.percentile(edge_depths, 10)
        corners = [
            pixel_to_3d(x1, y1, box_z),
            pixel_to_3d(x2, y1, box_z),
            pixel_to_3d(x2, y2, box_z),
            pixel_to_3d(x1, y2, box_z),
        ]
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        color = box_colors[i % len(box_colors)]
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
        geoms.append(line_set)

    return geoms


def _score_segment_for_match(seg_mask_local, split_mask, border_band, kernel3):
    """Bewertet ein Segment: border_ratio, closure, seg_mask_local."""
    seg_size = int(seg_mask_local.sum())
    if seg_size == 0:
        return None

    split_dilated = cv2.dilate(split_mask, kernel3, iterations=2)
    on_border = (seg_mask_local > 0) & border_band
    unexplained_border = on_border & (split_dilated == 0)
    border_ratio = int(unexplained_border.sum()) / seg_size

    dilated = cv2.dilate(seg_mask_local, kernel3, iterations=1)
    contour = (dilated.astype(bool)) & (~seg_mask_local.astype(bool))
    contour_pixels = int(contour.sum())
    if contour_pixels == 0:
        return None

    on_gradient = contour & (split_mask > 0)
    on_border_explained = contour & border_band & (split_dilated > 0)
    closed_pixels = int(np.sum(on_gradient | on_border_explained))
    closed_score = closed_pixels / contour_pixels

    return {
        "border_ratio": border_ratio,
        "closure": closed_score,
        "seg_mask_local": seg_mask_local,
        "size": seg_size,
    }


def _log_match_skip(label, reason, diagnostics=None, edge_pixels=None, min_edge=None):
    """Diagnostik für verworfene DINO+Gradient-Matches."""
    if diagnostics:
        best = max(diagnostics, key=lambda d: d["closure"])
        print(
            f"[MATCH] SKIP '{label}': {reason} | "
            f"best_seg={best['seg_id']} closure={best['closure']*100:.0f}% "
            f"border={best['border_ratio']*100:.1f}% size={best['size']}px"
        )
        for d in diagnostics:
            flag = "ok" if (
                d["border_ratio"] <= MATCH_BORDER_TOUCH_RATIO
                and d["closure"] >= MATCH_CLOSURE_RATIO
            ) else "  "
            print(
                f"  {flag} seg={d['seg_id']} closure={d['closure']*100:.0f}% "
                f"border={d['border_ratio']*100:.1f}% size={d['size']}px"
            )
    elif edge_pixels is not None:
        print(
            f"[MATCH] SKIP '{label}': {reason} "
            f"(kanten={edge_pixels}px, min={min_edge}px)"
        )
    else:
        print(f"[MATCH] SKIP '{label}': {reason}")


def _fuse_dino_gradient_match(depth, dino_box, label, image_h, image_w,
                              border_touch_ratio=None, border_margin=1,
                              closure_ratio=None, pallet_relative=False):
    """
    Akzeptiere ein Paket wenn DINO-Box + durchgängige Gradient-Kante zusammenpassen.
    Prüft alle Segmente und wählt das beste (höchste closure, border ok).
    """
    if border_touch_ratio is None:
        border_touch_ratio = MATCH_BORDER_TOUCH_RATIO
    if closure_ratio is None:
        closure_ratio = MATCH_CLOSURE_RATIO

    # Stufe 1: Grobe Kanten → Z-Ebene → ROI auf Slab → Matching in Z-Ebene
    analysis = analyze_box_gradient_z_aligned(depth, dino_box, pallet_relative=pallet_relative)
    x1, y1, x2, y2 = analysis["box_coords"]
    split_mask = analysis["split_mask"].astype(np.uint8)
    segment_labels = analysis["segment_labels"]
    z_slab_mask = analysis.get("z_slab_mask")
    used_z_slab = analysis.get("used_z_slab", False)

    if used_z_slab:
        print(
            f"[Z-SLAB] '{label}': DINO-ROI auf Gradient-Z-Ebene "
            f"z={analysis.get('z_plane_mm', 0):.0f}mm "
            f"tol=±{analysis.get('z_tolerance_mm', 0):.0f}mm "
            f"({int(z_slab_mask.sum()) if z_slab_mask is not None else 0}px)"
        )

    box_h_local, box_w_local = split_mask.shape
    box_area = max(box_h_local * box_w_local, 1)
    edge_pixels = int(split_mask.sum())

    if box_h_local < 10 or box_w_local < 10:
        _log_match_skip(label, "box zu klein")
        return None

    min_edge_pixels = max(40, int(box_area * 0.002))
    if edge_pixels < min_edge_pixels:
        _log_match_skip(
            label, "zu wenig Gradient-Kanten",
            edge_pixels=edge_pixels, min_edge=min_edge_pixels,
        )
        return None

    m = max(1, int(border_margin))
    border_band = np.zeros((box_h_local, box_w_local), dtype=bool)
    border_band[:m, :] = True
    border_band[-m:, :] = True
    border_band[:, :m] = True
    border_band[:, -m:] = True

    kernel3 = np.ones((3, 3), np.uint8)
    box_depth_local = depth[y1:y2, x1:x2]
    min_segment_pixels = max(200, int(box_area * 0.05))

    seg_ids = [s for s in np.unique(segment_labels) if s > 0]
    front_id = select_frontmost_segment(
        segment_labels, box_depth_local, pallet_relative=pallet_relative
    )

    diagnostics = []
    candidates = []

    for seg_id in seg_ids:
        seg_size = int(np.sum(segment_labels == seg_id))
        if seg_size < min_segment_pixels:
            continue

        seg_mask_local = (segment_labels == seg_id).astype(np.uint8)
        scored = _score_segment_for_match(seg_mask_local, split_mask, border_band, kernel3)
        if scored is None:
            continue

        entry = {
            "seg_id": int(seg_id),
            "size": seg_size,
            "border_ratio": scored["border_ratio"],
            "closure": scored["closure"],
            "seg_mask_local": scored["seg_mask_local"],
            "is_front": seg_id == front_id,
        }
        diagnostics.append(entry)

        if scored["border_ratio"] <= border_touch_ratio:
            candidates.append(entry)

    if not candidates:
        _log_match_skip(label, "alle Segmente occluded (Rand ohne Gradient)", diagnostics)
        return None

    # Bestes Segment: höchste closure unter border-ok; Front-Segment bei Gleichstand bevorzugen
    candidates.sort(
        key=lambda c: (c["closure"], c["is_front"], c["size"]),
        reverse=True,
    )
    best = candidates[0]

    if best["closure"] < closure_ratio:
        _log_match_skip(
            label,
            f"Geschlossenheit < {closure_ratio*100:.0f}%",
            diagnostics,
        )
        return None

    chosen = best
    seg_mask_local = chosen["seg_mask_local"]
    border_ratio = chosen["border_ratio"]
    closed_score = chosen["closure"]
    seg_id = chosen["seg_id"]

    print(
        f"[MATCH] OK '{label}': seg={seg_id} closure={closed_score*100:.0f}% "
        f"border={border_ratio*100:.1f}% (front={chosen['is_front']})"
    )

    # In globale Bildmaske einsetzen (2D-Segment vor Z-Schnitt)
    global_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    seg_ys_local, seg_xs_local = np.where(seg_mask_local > 0)
    seg_xs_global = seg_xs_local + x1
    seg_ys_global = seg_ys_local + y1
    valid = (seg_xs_global < image_w) & (seg_ys_global < image_h)
    seg_xs_global = seg_xs_global[valid]
    seg_ys_global = seg_ys_global[valid]
    if len(seg_xs_global) == 0:
        return None
    global_mask[seg_ys_global, seg_xs_global] = 1

    # Feinabstimmung: Maske auf Z-Slab (bereits vor Match) bzw. Vorderebene
    constraint_mask = None
    if z_slab_mask is not None and used_z_slab:
        constraint_mask = z_slab_mask
    else:
        constraint_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        front_local = analysis.get("front_layer_mask")
        if front_local is not None:
            constraint_mask[y1:y2, x1:x2] = front_local

    aligned_mask, z_stats = align_mask_to_depth_plane(
        global_mask, depth, split_mask, (x1, y1, x2, y2),
        front_layer_mask=constraint_mask,
    )
    if aligned_mask is None and z_stats.get("reject_reason") == "low_keep_ratio":
        aligned_mask, z_stats = align_mask_to_depth_plane(
            global_mask, depth, split_mask, (x1, y1, x2, y2),
            front_layer_mask=None,
            min_keep_ratio=Z_ALIGN_MIN_KEEP_RATIO * 0.7,
        )
    if aligned_mask is None:
        reason = z_stats.get("reject_reason", "unknown")
        print(
            f"[Z-ALIGN] REJECT '{label}': {reason} "
            f"(residual={z_stats.get('z_residual_mm', 0):.0f}mm, "
            f"kept={z_stats.get('pixels_kept_ratio', 0)*100:.0f}%, "
            f"tol={z_stats.get('tolerance_mm', 0):.0f}mm)"
        )
        return None

    seg_ys_a, seg_xs_a = np.where(aligned_mask > 0)
    if len(seg_xs_a) == 0:
        return None

    matched_box = [
        int(seg_xs_a.min()),
        int(seg_ys_a.min()),
        int(seg_xs_a.max()),
        int(seg_ys_a.max()),
    ]

    print(
        f"[Z-ALIGN] '{label}': z_plane={z_stats['z_plane_mm']:.0f}mm, "
        f"tol={z_stats['tolerance_mm']:.0f}mm, kept={z_stats['pixels_kept_ratio']*100:.0f}%, "
        f"residual={z_stats['z_residual_mm']:.0f}mm"
    )

    return {
        "label": label,
        "dino_box": [x1, y1, x2, y2],
        "matched_box": matched_box,
        "mask": aligned_mask,
        "analysis": analysis,
        "edge_pixels": edge_pixels,
        "segment_pixels": int(aligned_mask.sum()),
        "border_ratio": border_ratio,
        "closure": closed_score,
        "z_stats": z_stats,
        "used_z_slab": used_z_slab,
        "z_plane_mm": analysis.get("z_plane_mm"),
    }


def _match_z_plane_m(match):
    """Z-Ebene der Gradient-Kante in Metern (depth_rel: größer = näher/vorne)."""
    zs = match.get("z_stats") or {}
    z = zs.get("z_plane_m")
    if z is not None:
        return float(z)
    z_mm = match.get("z_plane_mm")
    if z_mm is not None:
        return float(z_mm) / 1000.0
    return float("inf")


def _mask_iou(mask_a, mask_b):
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return inter / union if union > 0 else 0.0


def _bbox_inter_areas(box_a, box_b):
    ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
    bx1, by1, bx2, by2 = [int(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter, area_a, area_b


def _bbox_iou(box_a, box_b):
    inter, area_a, area_b = _bbox_inter_areas(box_a, box_b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_containment(box_a, box_b):
    """Anteil der kleineren BBox, der vom Schnitt überdeckt wird."""
    inter, area_a, area_b = _bbox_inter_areas(box_a, box_b)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _mask_containment(mask_a, mask_b):
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    inter = int(np.sum(a & b))
    smaller = min(int(a.sum()), int(b.sum()))
    return inter / smaller if smaller > 0 else 0.0


def _match_overlap_metrics(m, k):
    """Liefert iou_mask, iou_bbox, contain_mask, contain_bbox.

    iou_bbox/contain_bbox = max(matched_box, dino_box) – DINO ist die ursprüngliche
    Region und oft die einzige, die Überlappung sichtbar macht.
    """
    iou_mask = _mask_iou(m["mask"], k["mask"])
    contain_mask = _mask_containment(m["mask"], k["mask"])
    iou_bbox = 0.0
    contain_bbox = 0.0
    if MATCH_DEDUP_USE_BBOX:
        box_m = m.get("matched_box", m["dino_box"])
        box_k = k.get("matched_box", k["dino_box"])
        iou_bbox = _bbox_iou(box_m, box_k)
        contain_bbox = _bbox_containment(box_m, box_k)
        if MATCH_DEDUP_USE_DINO_BBOX:
            dino_m = m.get("dino_box")
            dino_k = k.get("dino_box")
            if dino_m is not None and dino_k is not None:
                iou_bbox = max(iou_bbox, _bbox_iou(dino_m, dino_k))
                contain_bbox = max(contain_bbox, _bbox_containment(dino_m, dino_k))
    return iou_mask, iou_bbox, contain_mask, contain_bbox


def deduplicate_overlapping_matches(matches, iou_threshold=None, return_excluded=False):
    """
    Nach dem Matching: überlappende Pakete zusammenführen.
    Bei Überlappung bleibt das Paket mit der vorderen Gradient-Kante (größeres depth_rel).

    Args:
        return_excluded: wenn True, gibt (kept, excluded) zurück. Jedes excluded
            dict erhält zusätzliche Felder _status="excluded_by_dedup" und
            _occluded_by (label des überdeckenden Match).
    """
    if iou_threshold is None:
        iou_threshold = MATCH_DEDUP_IOU
    if len(matches) <= 1:
        return (matches, []) if return_excluded else matches

    prefer_closer = MATCH_DEDUP_KEEP_CLOSER
    sorted_matches = sorted(
        matches,
        key=_match_z_plane_m,
        reverse=not prefer_closer,
    )
    keep = []
    excluded = []

    for m in sorted_matches:
        overlaps_kept = False
        occluded_by = None
        for k in keep:
            iou_mask, iou_bbox, contain_mask, contain_bbox = _match_overlap_metrics(m, k)
            iou = max(iou_mask, iou_bbox) if MATCH_DEDUP_USE_BBOX else iou_mask
            containment = max(contain_mask, contain_bbox) if MATCH_DEDUP_USE_BBOX else contain_mask
            z_m = _match_z_plane_m(m) * 1000
            z_k = _match_z_plane_m(k) * 1000
            z_diff_m = abs(z_m - z_k) / 1000.0

            # Schwellen abhängig von Z-Differenz (mehr Δz → strenger gegen das hintere Paket)
            if z_diff_m >= MATCH_DEDUP_Z_DEEP_M:
                iou_thr = MATCH_DEDUP_IOU_DEEP
                cont_thr = MATCH_DEDUP_CONTAINMENT_DEEP
            elif z_diff_m >= MATCH_DEDUP_Z_OCCLUDE_M:
                iou_thr = MATCH_DEDUP_IOU_FAR
                cont_thr = MATCH_DEDUP_CONTAINMENT_FAR
            else:
                iou_thr = iou_threshold
                cont_thr = MATCH_DEDUP_CONTAINMENT

            triggers_iou = iou > iou_thr
            triggers_contain = containment > cont_thr

            if triggers_iou or triggers_contain:
                if z_diff_m < MATCH_DEDUP_Z_DIFF_M:
                    print(
                        f"[DEDUP] gleiche Ebene '{m['label']}' & '{k['label']}': "
                        f"|Δz|={z_diff_m*1000:.0f}mm < {MATCH_DEDUP_Z_DIFF_M*1000:.0f}mm → behalte beide"
                    )
                    continue
                trigger = "IoU" if triggers_iou else "Containment"
                print(
                    f"[DEDUP] '{m['label']}' verworfen: {trigger}={max(iou, containment):.2f} "
                    f"(iou_maske={iou_mask:.2f} iou_bbox={iou_bbox:.2f} "
                    f"cont_maske={contain_mask:.2f} cont_bbox={contain_bbox:.2f}, "
                    f"|Δz|={z_diff_m*1000:.0f}mm) "
                    f"mit '{k['label']}' (z={z_m:.0f}mm vs z={z_k:.0f}mm gewinnt)"
                )
                overlaps_kept = True
                occluded_by = k["label"]
                break
            elif iou > 0.02 or containment > 0.05:
                print(
                    f"[DEDUP] knapp: '{m['label']}' & '{k['label']}' "
                    f"iou={iou:.2f} cont={containment:.2f} |Δz|={z_diff_m*1000:.0f}mm "
                    f"(Schwellen IoU>{iou_thr} cont>{cont_thr})"
                )
        if overlaps_kept:
            m_excluded = dict(m)
            m_excluded["_status"] = "excluded_by_dedup"
            m_excluded["_occluded_by"] = occluded_by
            excluded.append(m_excluded)
        else:
            keep.append(m)

    removed = len(matches) - len(keep)
    if removed > 0:
        print(f"[DEDUP] {len(keep)}/{len(matches)} Matches nach Überlappungsfilter")
    if return_excluded:
        return keep, excluded
    return keep


def extract_dino_gradient_masks(session_path, dino_debug, sobel_viz_data,
                                closure_ratio=None, border_touch_ratio=None,
                                return_excluded=False):
    """
    Extrahiert Pakete, die in der DINO-Box durchgängig von Gradient-Kanten umrandet sind.
    Wird sowohl von der Visualisierung als auch von der Pipeline (→ SAM3D) verwendet.

    Returns:
        list[dict] mit Keys: label, dino_box, matched_box, mask (HxW), analysis,
                              edge_pixels, segment_pixels, border_ratio, closure
        Leere Liste wenn keine Daten/Matches.
    """
    empty: tuple = ([], []) if return_excluded else []

    if dino_debug is None or sobel_viz_data is None:
        return empty

    depth = sobel_viz_data.get("depth")
    if depth is None:
        return empty

    boxes = dino_debug.get("post_iou_boxes", [])
    labels = dino_debug.get("post_iou_labels", [])
    if not boxes:
        return empty

    if closure_ratio is None:
        closure_ratio = MATCH_CLOSURE_RATIO
    if border_touch_ratio is None:
        border_touch_ratio = MATCH_BORDER_TOUCH_RATIO

    pallet_relative = sobel_viz_data.get("session_context") is not None
    H, W = depth.shape
    matches = []
    for box, label in zip(boxes, labels):
        match = _fuse_dino_gradient_match(
            depth, box, label, H, W,
            border_touch_ratio=border_touch_ratio,
            closure_ratio=closure_ratio,
            pallet_relative=pallet_relative,
        )
        if match is not None:
            matches.append(match)

    if return_excluded:
        kept, excluded = deduplicate_overlapping_matches(matches, return_excluded=True)
        return kept, excluded
    matches = deduplicate_overlapping_matches(matches)
    return matches


def visualize_dino_boxes_with_gradient_edges(
    session_path, dino_debug, sobel_viz_data, window_name, matches=None, session_context=None
):
    """
    3D: Nur gematchte Paare (DINO + Gradienten-Analyse).
    Box-Rahmen und grüne Kanten nur wenn beide Signale in derselben Region vorliegen.
    """
    if dino_debug is None or sobel_viz_data is None:
        print("[VIZ] Keine DINO- oder Sobel-Daten für Box+Gradient-Ansicht.")
        return

    depth = sobel_viz_data.get("depth")
    if depth is None:
        print("[VIZ] Kein Tiefenbild in Sobel-Daten.")
        return

    boxes = dino_debug.get("post_iou_boxes", [])
    if not boxes:
        print("[VIZ] Keine DINO-Boxen (post_iou) vorhanden.")
        return

    if session_context is None and sobel_viz_data is not None:
        session_context = sobel_viz_data.get("session_context")
    all_points, _, H, W, base_colors, _ = _load_pcd_for_viz(session_path, session_context)

    if matches is None:
        matches = extract_dino_gradient_masks(session_path, dino_debug, sobel_viz_data)

    n_total = len(boxes)
    n_skipped = n_total - len(matches)

    if not matches:
        print("[VIZ] Keine DINO+Gradient-Matches – nichts zu zeichnen.")
        return

    box_colors = _generate_unique_colors(len(matches))
    colors = base_colors.copy()
    edge_count = 0
    mask_pixels = 0
    matched_boxes = []
    z_planes = []

    for i, match in enumerate(matches):
        analysis = match["analysis"]
        x1, y1, x2, y2 = analysis["box_coords"]
        split_mask = analysis["split_mask"]
        global_mask = match["mask"]
        z_stats = match.get("z_stats", {})
        z_plane = z_stats.get("z_plane_m")
        tol = z_stats.get("tolerance_m", 0.03)
        box_color = np.array(box_colors[i])
        matched_boxes.append(match["matched_box"])
        z_planes.append(z_plane)

        # Maske in Box-Farbe einfärben (Kanten und Innenfläche)
        mask_ys, mask_xs = np.where(global_mask > 0)
        for gy, gx in zip(mask_ys, mask_xs):
            colors[gy * W + gx] = box_color
        mask_pixels += len(mask_ys)

        # Gradient-Kanten in derselben Box-Farbe (nicht mehr grün)
        box_h, box_w = split_mask.shape
        for by in range(box_h):
            for bx in range(box_w):
                img_y = y1 + by
                img_x = x1 + bx
                if img_y >= H or img_x >= W:
                    continue
                if split_mask[by, bx] > 0 and depth[img_y, img_x] > 0:
                    if z_plane is None or abs(depth[img_y, img_x] - z_plane) <= tol:
                        colors[img_y * W + img_x] = box_color
                        edge_count += 1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geoms = [pcd]

    print(
        f"[VIZ] {len(matches)}/{n_total} gematcht "
        f"(Maske={mask_pixels}px, Kanten={edge_count}px), "
        f"{n_skipped} übersprungen (occluded oder Kante nicht durchgängig)"
    )
    for i, m in enumerate(matches):
        d = m["dino_box"]
        f = m["matched_box"]
        zs = m.get("z_stats", {})
        print(
            f"  Match {i}: '{m['label']}' "
            f"DINO=[{d[0]},{d[1]},{d[2]},{d[3]}] → "
            f"Maske=[{f[0]},{f[1]},{f[2]},{f[3]}] "
            f"(Segment={m['segment_pixels']}px, "
            f"Geschlossenheit={m['closure']*100:.1f}%, "
            f"Rand={m['border_ratio']*100:.1f}%, "
            f"z={zs.get('z_plane_mm', 0):.0f}mm)"
        )
    o3d.visualization.draw_geometries(geoms, window_name=window_name)


def visualize_sobel_edges(session_path, viz_data, window_title_suffix="", session_context=None):
    """
    3D Visualisierung: Gradienten/Kanten Analyse.
    Färbt die Punktwolke basierend auf der Gradienten-Magnitude.
    """
    if viz_data is None:
        print("[VIZ] Keine Sobel-Daten vorhanden.")
        return

    if session_context is None and viz_data is not None:
        session_context = viz_data.get("session_context")
    all_points, _, H, W, base_colors, _ = _load_pcd_for_viz(session_path, session_context)

    gradient = viz_data["gradient_magnitude"]
    edges = viz_data["edges"]
    
    # Gradienten normalisieren für Farb-Mapping (0-1)
    # Clip bei 50mm für Kontrast
    grad_norm = np.clip(gradient, 0, 50) / 50.0
    
    colors = base_colors.copy()
    grad_flat = grad_norm.flatten()
    colors[:, 0] = np.maximum(colors[:, 0], grad_flat)
    colors[:, 2] = np.minimum(colors[:, 2], 1 - grad_flat)

    edges_flat = edges.flatten()
    colors[edges_flat > 0] = [0, 1, 0]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    title = f"5/6: Gradienten/Spalten Analyse{window_title_suffix} (Blau=Flach, Rot=Steil, Grün=Kante)"
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_per_box_gradient(session_path, viz_data, dino_debug, session_context=None):
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
    
    if session_context is None and viz_data is not None:
        session_context = viz_data.get("session_context")
    all_points, _, H, W, pcd_colors_flat, workspace_mask = _load_pcd_for_viz(
        session_path, session_context
    )
    base_colors = pcd_colors_flat.reshape(H, W, 3)
    if workspace_mask is not None:
        base_colors[~workspace_mask] = [0.15, 0.15, 0.15]
    
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


def visualize_all_stage5_matches(
    session_path,
    kept: list,
    excluded: list,
    window_name: str = "Stage 5: Alle Matches",
    session_context=None,
):
    """
    3D-Ansicht aller Stage-5-Matches: kept (grün) + dedup-excluded (rot).

    Die exkludierten Matches sind Pakete, die durch den Overlap-Filter
    entfernt wurden – meistens das tiefere Paket bei einem Stack. Stage 8
    nutzt deren Top-Höhe, um Bounding-Boxen nach unten zu ziehen.
    """
    if not kept and not excluded:
        print("[VIZ] Stage-5: keine Matches – übersprungen.")
        return

    all_points, _, H, W, base_colors, _ = _load_pcd_for_viz(session_path, session_context)
    colors = base_colors.copy() * 0.55

    # Kept matches: grün-blaue Farbtöne, voll deckend
    for i, m in enumerate(kept):
        mask = np.asarray(m["mask"]) > 0
        if not mask.any():
            continue
        hue = (i * 47 + 90) / 360.0
        rgb = _hsv_to_rgb(hue, 0.85, 0.95)
        ys, xs = np.where(mask)
        idx = ys * W + xs
        colors[idx] = rgb

    # Excluded matches: rot-orange, leicht ausgegraut
    for i, m in enumerate(excluded):
        mask = np.asarray(m["mask"]) > 0
        if not mask.any():
            continue
        hue = (i * 23) / 360.0 % 1.0
        rgb = _hsv_to_rgb(hue, 0.95, 0.75)
        ys, xs = np.where(mask)
        idx = ys * W + xs
        colors[idx] = rgb

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(
        f"[VIZ] Stage-5: {len(kept)} kept (grün-blau), "
        f"{len(excluded)} excluded (rot-orange)"
    )
    for i, m in enumerate(kept):
        zs = m.get("z_stats") or {}
        print(
            f"  KEPT     #{i} '{m['label']}': z={zs.get('z_plane_mm', 0):.0f}mm  "
            f"closure={m.get('closure', 0)*100:.0f}%  bbox={m.get('matched_box')}"
        )
    for i, m in enumerate(excluded):
        zs = m.get("z_stats") or {}
        print(
            f"  EXCLUDED #{i} '{m['label']}': z={zs.get('z_plane_mm', 0):.0f}mm  "
            f"occluded_by='{m.get('_occluded_by', '?')}'  bbox={m.get('matched_box')}"
        )

    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def _hsv_to_rgb(h: float, s: float, v: float) -> list:
    """Pure-Python HSV->RGB (h,s,v in [0,1])."""
    import colorsys
    return list(colorsys.hsv_to_rgb(h, s, v))


def visualize_3d(session_path, refined_masks, refined_labels, sobel_viz_data=None,
                 original_masks=None, original_labels=None, dino_debug=None,
                 sam3d_masks=None, sam3d_labels=None,
                 closed_matches=None, excluded_matches=None,
                 session_context=None, candidates=None):
    """
    Hauptfunktion: Zeigt alle Debug-Visualisierungen nacheinander.

    Reihenfolge:
    1. RGBD mit Original-Farben
    2. DINO Raw Boxes (alle erkannten)
    3. DINO Boxes nach IoU-NMS
    4. Per-Box Gradient-Analyse (wo wurden Segmente getrennt?)
    5. Globale Gradient/Edges Analyse
    6. DINO-Boxen + Gradienten-Kanten (gematcht pro Box)
    7. Alle Stage-5-Matches inkl. dedup-excluded (Nachbar-Pool für Stage 8)
    8. SAM3D Segmente (eigener Output, optional)
    9. Bottom-Plane Inference (extrudierte OBBs, optional)
    """
    has_sam3d = sam3d_masks is not None and sam3d_labels is not None
    has_bottom = candidates is not None and len(candidates) > 0
    has_all_matches = bool(closed_matches) or bool(excluded_matches)
    total_steps = 6 + int(has_all_matches) + int(has_sam3d) + int(has_bottom)
    print(f"\n[VIZ] Starte {total_steps}-fache Debug-Visualisierung...")
    
    # 1. RGBD mit Original-Farben
    print("[VIZ] 1/6: RGBD Punktwolke...")
    visualize_3d_rgbd(session_path, session_context=session_context)

    if dino_debug:
        print("[VIZ] 2/6: DINO Raw Boxes...")
        visualize_dino_boxes(session_path, dino_debug, stage="raw", session_context=session_context)

    if dino_debug:
        print("[VIZ] 3/6: DINO Boxes nach IoU-NMS...")
        visualize_dino_boxes(
            session_path, dino_debug, stage="post_iou", session_context=session_context
        )

    if sobel_viz_data and "per_box_analysis" in sobel_viz_data:
        print("[VIZ] 4/6: Per-Box Gradient-Analyse...")
        visualize_per_box_gradient(
            session_path, sobel_viz_data, dino_debug, session_context=session_context
        )

    if sobel_viz_data:
        print("[VIZ] 5/6: Globale Gradient-Analyse...")
        visualize_sobel_edges(session_path, sobel_viz_data, session_context=session_context)

    if dino_debug and sobel_viz_data:
        print(f"[VIZ] 6/{total_steps}: DINO-Boxen + Gradient-Kanten...")
        visualize_dino_boxes_with_gradient_edges(
            session_path,
            dino_debug,
            sobel_viz_data,
            window_name=f"6/{total_steps}: Geschlossene Pakete (DINO ∩ durchgängige Kante)",
            matches=closed_matches,
            session_context=session_context,
        )

    next_step = 7
    if has_all_matches:
        step = next_step
        print(
            f"[VIZ] {step}/{total_steps}: Alle Stage-5-Matches "
            f"(kept={len(closed_matches or [])}, "
            f"excluded={len(excluded_matches or [])})..."
        )
        visualize_all_stage5_matches(
            session_path,
            kept=closed_matches or [],
            excluded=excluded_matches or [],
            window_name=f"{step}/{total_steps}: Alle erkannten Pakete (kept + dedup-excluded)",
            session_context=session_context,
        )
        next_step += 1

    if has_sam3d:
        step = next_step
        print(f"[VIZ] {step}/{total_steps}: SAM3D Segmente (Input = Stufe-6-Masken)...")
        visualize_3d_colored(
            session_path,
            sam3d_masks,
            sam3d_labels,
            window_name=f"{step}/{total_steps}: SAM3D Segmente (Input = geschlossene Pakete)",
            session_context=session_context,
        )
        next_step += 1

    if has_bottom:
        step = total_steps
        print(f"[VIZ] {step}/{total_steps}: Bottom-Plane Inference (extrudierte Volumen)...")
        visualize_bottom_inference_3d(
            session_path,
            candidates,
            window_name=f"{step}/{total_steps}: Bottom-Plane Inference (OBB + Bodenhöhe)",
            session_context=session_context,
        )
    
    print(f"[VIZ] Alle {total_steps} Visualisierungen abgeschlossen.\n")
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
