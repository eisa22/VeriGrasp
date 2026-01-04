#!/usr/bin/env python3
"""
Test-Script: Vollständiger DINO + SAM + 3D Pipeline
Zeigt den kompletten Flow mit korrekter Farbgebung pro Objekt.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import open3d as o3d
import os
from config import *
from GroundingSAM.grounding_sam import run_grounding_sam
from Sam3D.sam3d import SAM3D


def visualize_2d(orig_image, boxes, masks, labels, scores):
    """2D Visualisierung mit farbigen Boxen und Masken."""
    img = orig_image.copy()
    draw = ImageDraw.Draw(img)
    
    # Feste Farben für bessere Unterscheidbarkeit
    colors = [
        (255, 0, 0),    # Rot
        (0, 255, 0),    # Grün
        (0, 0, 255),    # Blau
        (255, 255, 0),  # Gelb
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Violett
    ]
    
    print(f"\n{'='*60}")
    print(f"2D VISUALISIERUNG")
    print(f"{'='*60}")
    
    for i, (box, mask, label, score) in enumerate(zip(boxes, masks, labels, scores)):
        c = colors[i % len(colors)]
        score_val = score.item() if torch.is_tensor(score) else score
        
        # Bounding Box
        x0, y0, x1, y1 = [int(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=c, width=3)
        draw.text((x0, max(0, y0-14)), f"{label} ({score_val:.2f})", fill=c)
        
        # Maske
        mask_pixels = mask.sum()
        color_names = ["Rot", "Grün", "Blau", "Gelb", "Magenta", "Cyan", "Orange", "Violett"]
        print(f"  Objekt {i}: '{label}' → {color_names[i % len(color_names)]} ({mask_pixels} Pixel)")
        
        # Semi-transparente Maske
        color_layer = np.zeros((*mask.shape, 3), dtype=np.uint8)
        color_layer[...] = c
        alpha = (mask * 100).astype(np.uint8)
        img.paste(Image.fromarray(color_layer), mask=Image.fromarray(alpha))
    
    print(f"\n→ Zeige 2D Bild...")
    img.show(title="2D: DINO + SAM")
    return img


def visualize_3d_custom(session_path, masks, labels):
    """
    Eigene 3D Visualisierung mit garantiert unterschiedlichen Farben.
    """
    print(f"\n{'='*60}")
    print(f"3D VISUALISIERUNG (Custom)")
    print(f"{'='*60}")
    
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
    all_colors = rgb.reshape(-1, 3) / 255.0
    
    # Transformation (wie in SAM3D)
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
        [1.0, 0.0, 0.0],  # Rot
        [0.0, 1.0, 0.0],  # Grün
        [0.0, 0.0, 1.0],  # Blau
        [1.0, 1.0, 0.0],  # Gelb
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Violett
    ]
    color_names = ["Rot", "Grün", "Blau", "Gelb", "Magenta", "Cyan", "Orange", "Violett"]
    
    # Erstelle Zuordnungsmaske (verhindert Überlappung)
    assignment = np.full((H, W), -1, dtype=np.int32)
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask_np = np.asarray(mask)
        
        # Nur Pixel verwenden, die noch keinem Objekt zugeordnet sind
        ys, xs = np.where((mask_np > 0) & (assignment == -1))
        
        if len(xs) == 0:
            print(f"  Objekt {i} '{label}': Alle Pixel bereits zugeordnet")
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
        print(f"  Objekt {i} '{label}': {len(segment_points)} Punkte → {color_names[i % len(color_names)]}")
    
    print(f"\n→ Öffne Open3D Viewer...")
    print(f"  (Schließen Sie das Fenster um fortzufahren)")
    o3d.visualization.draw_geometries(geoms, window_name="3D: Segmentierte Objekte")


def split_mask_by_depth_gaps(mask, depth_map, min_segment_size=500):
    """
    Teilt eine große Maske anhand von Tiefen-Kanten (Gradienten) auf.
    
    Kanten = Bereiche mit starkem Tiefengradienten (Spalten zwischen Paketen).
    """
    import cv2
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
    # Verwende Perzentil statt festen Threshold
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
    - Entfernt große Masken die mit kleineren überlappen (kleinere sind präziser)
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
            
            # Große Maske enthält kleine → entferne große
            # (kleine Masken sind präziser)
            smaller_in_larger = overlap / min(mask_sizes[i], mask_sizes[j])
            
            if smaller_in_larger > 0.8:  # 80% der kleineren Maske überlappt
                if mask_sizes[i] > mask_sizes[j]:
                    # i ist größer, j ist kleiner und fast komplett in i enthalten
                    # Aber wir behalten beide erstmal, da die große Maske noch andere Bereiche haben könnte
                    pass
                else:
                    pass
    
    # Spezialfall: Sehr große Masken (> 30% des Bildes) die mit mehreren kleinen überlappen
    total_pixels = masks[0].shape[0] * masks[0].shape[1]
    for i in range(n):
        if i in to_remove:
            continue
        
        if mask_sizes[i] > 0.3 * total_pixels:  # Maske > 30% des Bildes
            overlap_count = 0
            for j in range(n):
                if i == j or j in to_remove:
                    continue
                overlap = np.sum((masks[i] > 0) & (masks[j] > 0))
                if overlap > 0.5 * mask_sizes[j]:  # > 50% der kleinen Maske überlappt
                    overlap_count += 1
            
            if overlap_count >= 2:  # Überlappt mit mindestens 2 anderen Masken
                to_remove.add(i)
                print(f"  [FILTER] Maske {i} '{labels[i]}' entfernt: Große Maske ({mask_sizes[i]} Pixel) überlappt mit {overlap_count} kleineren")
    
    # Gefilterte Listen erstellen
    filtered_boxes = [b for idx, b in enumerate(boxes) if idx not in to_remove]
    filtered_masks = [m for idx, m in enumerate(masks) if idx not in to_remove]
    filtered_scores = [s for idx, s in enumerate(scores) if idx not in to_remove]
    filtered_labels = [l for idx, l in enumerate(labels) if idx not in to_remove]
    
    return filtered_boxes, filtered_masks, filtered_scores, filtered_labels


def main():
    print("="*60)
    print("VOLLSTÄNDIGER PIPELINE TEST")
    print("DINO → SAM → 2D/3D Visualisierung")
    print("="*60)
    
    session_path = BASE_PATH
    print(f"Session: {session_path}\n")
    
    # -------------------------------------------------------------------------
    # Phase 1: DINO + SAM
    # -------------------------------------------------------------------------
    print(f"{'='*60}")
    print("PHASE 1: Grounding DINO + SAM")
    print(f"{'='*60}")
    
    boxes, masks, scores, labels = run_grounding_sam(session_path)
    
    if len(boxes) == 0:
        print("\n[ERROR] Keine Objekte erkannt")
        return
    
    # -------------------------------------------------------------------------
    # Phase 1.5: Filter überlappende/doppelte Masken
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 1.5: Filter überlappende Masken")
    print(f"{'='*60}")
    
    original_count = len(masks)
    boxes, masks, scores, labels = filter_overlapping_masks(boxes, masks, scores, labels)
    print(f"\n→ {original_count} → {len(masks)} Masken (gefiltert: {original_count - len(masks)})")
    
    # -------------------------------------------------------------------------
    # Phase 1.6: Split große Masken anhand von Depth-Gaps
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 1.6: Split große Masken (Depth-Gap Detection)")
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
                # Keine Aufteilung möglich, behalte Original
                new_boxes.append(box)
                new_masks.append(mask)
                new_scores.append(score)
                new_labels.append(label)
        else:
            # Kleine Masken unverändert übernehmen
            new_boxes.append(box)
            new_masks.append(mask)
            new_scores.append(score)
            new_labels.append(label)
    
    boxes, masks, scores, labels = new_boxes, new_masks, new_scores, new_labels
    print(f"\n→ Nach Gap-Split: {len(masks)} Masken")
    
    print(f"\n→ Erkannt: {len(boxes)} Objekte")
    for i, (label, score) in enumerate(zip(labels, scores)):
        score_val = score.item() if torch.is_tensor(score) else score
        print(f"   [{i}] '{label}' (Score={score_val:.3f})")
    
    # -------------------------------------------------------------------------
    # Phase 2: 2D Visualisierung
    # -------------------------------------------------------------------------
    rgb_path = os.path.join(session_path, "rgb", "rgb_0000.png")
    orig_image = Image.open(rgb_path).convert("RGB")
    
    visualize_2d(orig_image, boxes, masks, labels, scores)
    
    # -------------------------------------------------------------------------
    # Phase 3: 3D Visualisierung (Custom - ohne SAM3D.process)
    # -------------------------------------------------------------------------
    visualize_3d_custom(session_path, masks, labels)
    
    # -------------------------------------------------------------------------
    # Zusammenfassung
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    print(f"DINO erkannte: {len(boxes)} Bounding Boxes")
    print(f"SAM segmentierte: {len(masks)} Masken")
    print(f"3D Punktwolken: {len(masks)} Objekte")
    
    # Überlappungsanalyse
    print(f"\nÜBERLAPPUNGS-ANALYSE:")
    total_pixels = masks[0].shape[0] * masks[0].shape[1]
    for i, (mask_i, label_i) in enumerate(zip(masks, labels)):
        for j, (mask_j, label_j) in enumerate(zip(masks, labels)):
            if i >= j:
                continue
            overlap = np.sum((mask_i > 0) & (mask_j > 0))
            if overlap > 0:
                print(f"  [{i}] '{label_i}' ∩ [{j}] '{label_j}': {overlap} Pixel überlappen")
    
    print(f"{'='*60}")
    print("✓ Fertig!")


if __name__ == "__main__":
    main()

