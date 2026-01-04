# main.py
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import open3d as o3d
import os
from GroundingSAM.grounding_sam import run_grounding_sam
from Sam3D.sam3d import SAM3D
from path_utils import get_all_session_paths
from config import DEBUG


def split_mask_by_depth_gaps(mask, depth_map, min_segment_size=500):
    """
    Teilt eine große Maske anhand von Tiefen-Kanten (Gradienten) auf.
    
    Kanten = Bereiche mit starkem Tiefengradienten (Spalten zwischen Paketen).
    """
    from scipy import ndimage
    
    H, W = depth_map.shape
    
    # Nur Tiefenwerte innerhalb der Maske
    masked_depth = depth_map.copy().astype(np.float32)
    masked_depth[mask == 0] = 0
    
    # Finde valide Bereiche
    valid = (mask > 0) & (depth_map > 0)
    if not valid.any():
        return [mask]
    
    valid_depths = masked_depth[valid]
    depth_min = valid_depths.min()
    depth_max = valid_depths.max()
    depth_range = depth_max - depth_min
    
    print(f"    [GAP] Depth Range: {depth_min:.3f}m - {depth_max:.3f}m (Δ={depth_range:.3f}m)")
    
    # Wenn kaum Variation, keine Trennung möglich
    if depth_range < 0.03:  # < 3cm Range
        print(f"    [GAP] Keine signifikante Tiefenvariation")
        return [mask]
    
    # Normalisiere Depth für Gradient-Berechnung
    depth_norm = np.zeros((H, W), dtype=np.float32)
    depth_norm[valid] = (masked_depth[valid] - depth_min) / depth_range
    
    # Berechne Gradient (Sobel)
    grad_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=5)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Nur Gradient innerhalb der Maske
    gradient[mask == 0] = 0
    
    # Finde Kanten mit hohem Gradient
    valid_gradients = gradient[valid]
    if len(valid_gradients) == 0:
        return [mask]
    
    edge_threshold = np.percentile(valid_gradients, 85)  # Top 15% sind Kanten
    
    print(f"    [GAP] Gradient-Threshold (85. Perzentil): {edge_threshold:.4f}")
    
    # Erstelle Kanten-Maske
    edges = (gradient > edge_threshold).astype(np.uint8) * 255
    
    # Erweitere Kanten um sichere Trennung zu gewährleisten
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Objekt-Maske = Original-Maske minus Kanten
    object_mask = mask.copy().astype(np.uint8) * 255
    object_mask[edges > 0] = 0
    
    # Morphologische Operationen
    kernel = np.ones((3, 3), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    
    # Connected Components
    n_labels, labels = cv2.connectedComponents(object_mask)
    
    print(f"    [GAP] → {n_labels - 1} Regionen gefunden")
    
    if n_labels <= 2:  # Nur 1 Region (+ Hintergrund)
        return [mask]
    
    # Erstelle separate Masken für jede Region
    result_masks = []
    for label_id in range(1, n_labels):
        segment = (labels == label_id).astype(np.uint8)
        pixel_count = segment.sum()
        
        if pixel_count < min_segment_size:
            print(f"    [GAP] Region {label_id}: {pixel_count} Pixel → zu klein")
            continue
        
        result_masks.append(segment)
        print(f"    [GAP] Region {label_id}: {pixel_count} Pixel → akzeptiert")
    
    if len(result_masks) == 0:
        return [mask]
    
    return result_masks


def filter_overlapping_masks(boxes, masks, scores, labels):
    """
    Filtert überlappende Masken:
    - Entfernt große Masken die mit kleineren überlappen
    - Entfernt fast identische Masken (Duplikate)
    """
    n = len(masks)
    if n <= 1:
        return boxes, masks, scores, labels
    
    # Berechne Maskengrößen
    mask_sizes = [np.sum(m) for m in masks]
    
    # Markiere Masken zum Entfernen
    to_remove = set()
    
    for i in range(n):
        if i in to_remove:
            continue
            
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            
            # Berechne Überlappung
            overlap = np.sum((masks[i] > 0) & (masks[j] > 0))
            
            if overlap == 0:
                continue
            
            # IoU berechnen
            union = np.sum((masks[i] > 0) | (masks[j] > 0))
            iou = overlap / union if union > 0 else 0
            
            # Fast identische Masken (IoU > 0.9) → entferne die mit niedrigerem Score
            if iou > 0.9:
                score_i = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]
                score_j = scores[j].item() if torch.is_tensor(scores[j]) else scores[j]
                if score_i >= score_j:
                    to_remove.add(j)
                    print(f"  [FILTER] Maske {j} '{labels[j]}' entfernt: Duplikat von {i} (IoU={iou:.2f})")
                else:
                    to_remove.add(i)
                    print(f"  [FILTER] Maske {i} '{labels[i]}' entfernt: Duplikat von {j} (IoU={iou:.2f})")
                continue
    
    # Spezialfall: Sehr große Masken (> 30% des Bildes) die mit mehreren kleinen überlappen
    total_pixels = masks[0].shape[0] * masks[0].shape[1]
    for i in range(n):
        if i in to_remove:
            continue
        
        if mask_sizes[i] > 0.3 * total_pixels:
            overlap_count = 0
            for j in range(n):
                if i == j or j in to_remove:
                    continue
                overlap = np.sum((masks[i] > 0) & (masks[j] > 0))
                if overlap > 0.5 * mask_sizes[j]:
                    overlap_count += 1
            
            if overlap_count >= 2:
                to_remove.add(i)
                print(f"  [FILTER] Maske {i} '{labels[i]}' entfernt: Große Maske ({mask_sizes[i]} Pixel) überlappt mit {overlap_count} kleineren")
    
    # Gefilterte Listen erstellen
    filtered_boxes = [b for idx, b in enumerate(boxes) if idx not in to_remove]
    filtered_masks = [m for idx, m in enumerate(masks) if idx not in to_remove]
    filtered_scores = [s for idx, s in enumerate(scores) if idx not in to_remove]
    filtered_labels = [l for idx, l in enumerate(labels) if idx not in to_remove]
    
    return filtered_boxes, filtered_masks, filtered_scores, filtered_labels


def visualize_2d(orig_image, boxes, masks, labels, scores):
    """2D Visualisierung mit farbigen Boxen und Masken."""
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
    """3D Visualisierung mit farbigen Segmenten."""
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
    
    # Transformation
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


def main():
    # Alle Sessions holen
    all_sessions = get_all_session_paths()
    print(f"[MAIN] Gefundene Sessions: {len(all_sessions)}")
    
    for session_path in all_sessions:
        session_name = os.path.basename(session_path)
        print(f"\n{'='*60}")
        print(f"[MAIN] Verarbeite Session: {session_name}")
        print(f"{'='*60}")
        
        # -------------------------------------------------------------------------
        # Phase 1: DINO + SAM
        # -------------------------------------------------------------------------
        boxes, masks, scores, labels = run_grounding_sam(session_path)
        
        if len(masks) == 0:
            print(f"[MAIN] {session_name}: Keine Masken gefunden, überspringe Session.")
            continue
        
        # -------------------------------------------------------------------------
        # Phase 2: Filter überlappende/doppelte Masken
        # -------------------------------------------------------------------------
        print(f"\n{'='*60}")
        print("FILTER: Überlappende Masken")
        print(f"{'='*60}")
        
        original_count = len(masks)
        boxes, masks, scores, labels = filter_overlapping_masks(boxes, masks, scores, labels)
        print(f"\n→ {original_count} → {len(masks)} Masken (gefiltert: {original_count - len(masks)})")
        
        # -------------------------------------------------------------------------
        # Phase 3: Split große Masken anhand von Depth-Gaps
        # -------------------------------------------------------------------------
        print(f"\n{'='*60}")
        print("SPLIT: Große Masken (Depth-Gap Detection)")
        print(f"{'='*60}")
        
        # Lade Depth Map
        depth_path = os.path.join(session_path, "distance_to_image_plane",
                                   "distance_to_image_plane_0000.npy")
        depth_map = np.load(depth_path)
        
        total_pixels = masks[0].shape[0] * masks[0].shape[1]
        
        new_boxes = []
        new_masks = []
        new_scores = []
        new_labels = []
        
        for i, (box, mask, score, label) in enumerate(zip(boxes, masks, scores, labels)):
            mask_size = np.sum(mask)
            
            # Nur große Masken splitten (> 15% des Bildes)
            if mask_size > 0.15 * total_pixels:
                print(f"\nMaske {i} '{label}': {mask_size} Pixel (groß → versuche Split)")
                
                split_masks = split_mask_by_depth_gaps(mask, depth_map, min_segment_size=500)
                
                if len(split_masks) > 1:
                    print(f"  → Aufgeteilt in {len(split_masks)} Segmente!")
                    for j, split_mask in enumerate(split_masks):
                        # Berechne neue Bounding Box
                        ys, xs = np.where(split_mask > 0)
                        if len(xs) > 0:
                            new_box = [float(xs.min()), float(ys.min()), 
                                       float(xs.max()), float(ys.max())]
                            new_boxes.append(new_box)
                            new_masks.append(split_mask)
                            new_scores.append(score)
                            new_labels.append(f"{label}_{j+1}")
                else:
                    new_boxes.append(box)
                    new_masks.append(mask)
                    new_scores.append(score)
                    new_labels.append(label)
            else:
                new_boxes.append(box)
                new_masks.append(mask)
                new_scores.append(score)
                new_labels.append(label)
        
        boxes, masks, scores, labels = new_boxes, new_masks, new_scores, new_labels
        print(f"\n→ Nach Gap-Split: {len(masks)} Masken")
        
        print(f"\n[MAIN] {session_name}: {len(masks)} finale Objekte erkannt.")
        
        # -------------------------------------------------------------------------
        # Visualisierung (nur wenn DEBUG=True)
        # -------------------------------------------------------------------------
        if DEBUG:
            rgb_path = os.path.join(session_path, "rgb", "rgb_0000.png")
            orig_image = Image.open(rgb_path).convert("RGB")
            
            print(f"\n{'='*60}")
            print("VISUALISIERUNG")
            print(f"{'='*60}")
            visualize_2d(orig_image, boxes, masks, labels, scores)
            visualize_3d(session_path, masks, labels)
    
    print(f"\n{'='*60}")
    print(f"[MAIN] Alle {len(all_sessions)} Sessions verarbeitet!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
