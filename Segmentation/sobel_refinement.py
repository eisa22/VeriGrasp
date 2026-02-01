
import cv2
import numpy as np
import os
from PIL import Image
from scipy import ndimage


def analyze_box_gradient(depth, box, threshold_mm=15.0):
    """
    Analysiert den Gradienten innerhalb einer einzelnen Bounding Box.
    Erkennt signifikante Tiefensprünge und gibt potenzielle Trennlinien zurück.
    
    Args:
        depth: 2D Tiefenbild (float, in Metern)
        box: [x1, y1, x2, y2] Koordinaten der Box
        threshold_mm: Schwellwert für Tiefensprünge in mm
        
    Returns:
        dict mit:
        - 'split_mask': Binäre Maske der erkannten Trennlinien
        - 'num_segments': Anzahl der gefundenen Segmente
        - 'segment_labels': Label-Map der Segmente
        - 'gradient_magnitude': Gradient-Magnitude innerhalb der Box
    """
    x1, y1, x2, y2 = [int(c) for c in box]
    
    # Sichere Grenzen
    H, W = depth.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    
    # Box-Region extrahieren
    box_depth = depth[y1:y2, x1:x2].copy()
    
    if box_depth.size == 0:
        return {
            'split_mask': np.zeros((y2-y1, x2-x1), dtype=np.uint8),
            'num_segments': 1,
            'segment_labels': np.ones((y2-y1, x2-x1), dtype=np.int32),
            'gradient_magnitude': np.zeros((y2-y1, x2-x1))
        }
    
    # In mm konvertieren
    box_depth_mm = box_depth * 1000.0
    
    # Leichter Blur
    box_depth_mm = cv2.GaussianBlur(box_depth_mm, (3, 3), 0)
    
    # Sobel Gradienten berechnen
    grad_x = cv2.Sobel(box_depth_mm, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(box_depth_mm, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold: Signifikante Tiefensprünge
    split_mask = (gradient_magnitude > threshold_mm).astype(np.uint8)
    
    # Morphologische Operationen um Rauschen zu entfernen
    kernel = np.ones((3, 3), np.uint8)
    split_mask = cv2.morphologyEx(split_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    split_mask = cv2.morphologyEx(split_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Connected Components nach dem Split
    inverted = 1 - split_mask
    num_labels, labels = cv2.connectedComponents(inverted.astype(np.uint8))
    
    # Kleine Segmente ignorieren (weniger als 5% der Box)
    min_segment_pixels = (x2-x1) * (y2-y1) * 0.05
    for i in range(1, num_labels):
        if np.sum(labels == i) < min_segment_pixels:
            labels[labels == i] = 0
    
    # Labels neu nummerieren
    unique_labels = np.unique(labels[labels > 0])
    final_num_segments = len(unique_labels)
    
    return {
        'split_mask': split_mask,
        'num_segments': final_num_segments,
        'segment_labels': labels,
        'gradient_magnitude': gradient_magnitude,
        'box_coords': (x1, y1, x2, y2)
    }


def split_mask_by_gradient(mask, depth, box, min_segment_pixels=500):
    """
    Teilt eine Maske anhand von Gradientenanalyse in mehrere Segmente auf.
    
    Args:
        mask: Binäre Eingabemaske (2D numpy array)
        depth: Vollständiges Tiefenbild
        box: [x1, y1, x2, y2] der Box
        min_segment_pixels: Mindestgröße für ein Segment
    
    Returns:
        list: Liste von getrennten Masken
    """
    analysis = analyze_box_gradient(depth, box, threshold_mm=12.0)
    
    if analysis['num_segments'] <= 1:
        return [mask]
    
    x1, y1, x2, y2 = analysis['box_coords']
    segment_labels = analysis['segment_labels']
    
    # Erstelle individuelle Masken
    result_masks = []
    unique_labels = np.unique(segment_labels[segment_labels > 0])
    
    for seg_id in unique_labels:
        # Segment-Maske innerhalb der Box
        seg_mask_box = (segment_labels == seg_id).astype(np.uint8)
        
        # Zurück auf Vollbildgröße mappen
        full_seg_mask = np.zeros_like(mask, dtype=np.uint8)
        full_seg_mask[y1:y2, x1:x2] = seg_mask_box
        
        # Mit Original-Maske kombinieren (nur Pixel die auch in der Maske sind)
        combined = full_seg_mask & mask.astype(np.uint8)
        
        if np.sum(combined) >= min_segment_pixels:
            result_masks.append(combined)
    
    return result_masks if result_masks else [mask]


def apply_sobel_refinement(session_path, masks, labels, boxes=None):
    """
    Führt eine Gradientenanalyse mit Sobel auf dem Tiefenbild durch.
    NEU: Analysiert jede DINO-Box separat und trennt ggf. mehrere Pakete.
    
    Ziele:
    1. Erkennung von Spalten/Lücken (starke Tiefen-Sprünge).
    2. Präzisere Kanten.
    3. Filterung von verdeckten (occluded) Objekten.
    4. NEU: Automatische Trennung überlappender Pakete basierend auf Gradient.
    
    Args:
        session_path: Pfad zur Session (für Depth Image)
        masks: Liste der SAM-Masken
        labels: Liste der Labels
        boxes: Optional, Liste der Bounding Boxes von DINO
        
    Returns:
        tuple: (refined_masks, refined_labels, visualization_data)
    """
    print("\n[SOBEL] Starte Gradientenanalyse...")
    
    # 1. Lade Tiefenbild
    depth_path = os.path.join(session_path, "distance_to_image_plane", 
                               "distance_to_image_plane_0000.npy")
    if not os.path.exists(depth_path):
        print("[ERROR] Depth map not found.")
        return masks, labels, None
        
    depth = np.load(depth_path)
    H, W = depth.shape
    
    # 2. Globale Kanten-Erkennung für Visualisierung
    depth_mm = depth * 1000.0
    depth_mm = cv2.GaussianBlur(depth_mm, (3, 3), 0)
    norm_depth = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Canny Edge Detection
    binary_edges = cv2.Canny(norm_depth, 30, 100)
    
    # Sobel für Heatmap
    grad_x = cv2.Sobel(depth_mm, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_mm, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 3. Pro-Box Gradient-Analyse
    per_box_analysis = []
    if boxes:
        print(f"[SOBEL] Analysiere {len(boxes)} Boxes einzeln...")
        for i, box in enumerate(boxes):
            analysis = analyze_box_gradient(depth, box, threshold_mm=12.0)
            per_box_analysis.append(analysis)
            print(f"  Box {i}: {analysis['num_segments']} Segment(e) gefunden")
    
    # 4. Analyse & Refinement der Masken
    refined_masks = []
    refined_labels = []
    
    # Globale Min-Tiefe finden (oberste Ebene)
    valid_depths = depth[depth > 0]
    min_scene_depth = np.percentile(valid_depths, 1) if len(valid_depths) > 0 else 0
    
    print(f"[SOBEL] Min Scene Depth: {min_scene_depth*1000:.1f}mm")
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask_np = np.asarray(mask) > 0
        if mask_np.sum() == 0:
            continue
            
        # A. Occlusion Check
        obj_depths = depth[mask_np]
        obj_mean_depth = np.mean(obj_depths)
        
        OCCLUSION_MARGIN = 0.20  # 20cm Toleranz
        
        if obj_mean_depth > (min_scene_depth + OCCLUSION_MARGIN):
            print(f"  [FILTER] Maske {i} '{label}' entfernt: Zu tief/verdeckt (Z={obj_mean_depth*1000:.0f}mm vs Top={min_scene_depth*1000:.0f}mm)")
            continue
        
        # B. NEU: Gradientenbasierte Trennung
        if boxes and i < len(boxes):
            box = boxes[i]
            split_masks = split_mask_by_gradient(mask_np.astype(np.uint8), depth, box)
            
            if len(split_masks) > 1:
                print(f"  [SPLIT] Maske {i} '{label}' wurde in {len(split_masks)} Teile getrennt!")
                for j, sm in enumerate(split_masks):
                    refined_masks.append(sm)
                    refined_labels.append(f"{label}_{j+1}")
                continue
        
        # C. Standard Kanten-Check & Reclaim (wenn kein Split)
        internal_edges = binary_edges & mask_np
        edge_pixel_count = np.sum(internal_edges)
        
        if edge_pixel_count > 0:
            print(f"  [REFINE] Maske {i} '{label}': {edge_pixel_count} Edge-Pixel gefunden. Optimiere Ränder...")
            
            # Split: Ziehe Edge-Pixel ab
            core_mask = mask_np & (~binary_edges.astype(bool))
            
            # Morphologie (Clean up noise)
            kernel = np.ones((3,3), np.uint8)
            core_mask_u8 = core_mask.astype(np.uint8)
            core_mask_u8 = cv2.morphologyEx(core_mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Select: Finde Center Component
            num_labels, labels_im = cv2.connectedComponents(core_mask_u8)
            
            selected_core = None
            
            if num_labels > 2:
                 print(f"    -> Maske wurde durch Kanten in {num_labels-1} Teile gespalten!")
                 
                 # Bestimme Zentrum
                 M = cv2.moments(mask_np.astype(np.uint8))
                 if M["m00"] != 0:
                     cx = int(M["m10"] / M["m00"])
                     cy = int(M["m01"] / M["m00"])
                 else:
                     ys, xs = np.where(mask_np)
                     if len(xs) > 0:
                         cx = int(np.mean(xs))
                         cy = int(np.mean(ys))
                     else:
                         cx, cy = 0, 0
                 
                 found_center = False
                 for comp_idx in range(1, num_labels):
                     comp_mask = (labels_im == comp_idx).astype(np.uint8)
                     if comp_mask[cy, cx] > 0:
                         if comp_mask.sum() > 200:
                             selected_core = comp_mask
                             found_center = True
                         break
                 
                 if not found_center:
                     # Fallback: Largest
                     max_u = 0
                     for comp_idx in range(1, num_labels):
                        c_m = (labels_im == comp_idx).astype(np.uint8)
                        if c_m.sum() > max_u:
                            max_u = c_m.sum()
                            selected_core = c_m
            else:
                # Nur 1 Component (oder 0)
                if core_mask_u8.sum() > 200:
                    selected_core = core_mask_u8
            
            if selected_core is not None:
                # Watershed Refinement
                print(f"    -> Starte Watershed-Wachstum vom Kern aus...")
                
                markers = np.zeros_like(mask_np, dtype=np.int32)
                markers[selected_core == 1] = 2
                sure_bg = 1 - mask_np.astype(np.uint8)
                markers[sure_bg == 1] = 1
                
                grad_u8 = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                grad_bgr = cv2.cvtColor(grad_u8, cv2.COLOR_GRAY2BGR)
                
                cv2.watershed(grad_bgr, markers)
                
                final_mask = (markers == 2).astype(np.uint8)
                
                # Boundary inklusion
                boundaries = (markers == -1).astype(np.uint8)
                dilated_obj = cv2.dilate(final_mask, np.ones((3,3), np.uint8), iterations=1)
                valid_boundaries = boundaries & dilated_obj
                final_mask = final_mask | valid_boundaries
                final_mask = final_mask & mask_np.astype(np.uint8)
                
                refined_masks.append(final_mask)
                refined_labels.append(label)
            else:
                refined_masks.append(mask)
                refined_labels.append(label)
        else:
             refined_masks.append(mask)
             refined_labels.append(label)
            
    print(f"[SOBEL] Abgeschlossen. {len(masks)} -> {len(refined_masks)} Masken.")
    
    # Daten für Visualisierung
    viz_data = {
        "gradient_magnitude": gradient_magnitude,
        "edges": binary_edges,
        "depth": depth,
        "per_box_analysis": per_box_analysis  # NEU: Pro-Box Analyse
    }
    
    return refined_masks, refined_labels, viz_data
