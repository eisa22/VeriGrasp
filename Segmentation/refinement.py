"""
Segmentation/refinement.py
Modul für Segmentierungs-Verfeinerung basierend auf Tiefendaten.
"""
import numpy as np
import cv2


def split_mask_by_depth_gaps(mask, depth_map, min_segment_size=500):
    """
    Teilt eine große Maske anhand von Tiefen-Kanten (Gradienten) auf.
    
    Args:
        mask: Binäre Maske des zu teilenden Objekts
        depth_map: Tiefenkarte (in Metern)
        min_segment_size: Minimale Pixelanzahl pro Segment
        
    Returns:
        list: Liste von Segmentmasken (mindestens die ursprüngliche Maske)
    """
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
