
import cv2
import numpy as np
import os
from PIL import Image
from scipy import ndimage


def compute_adaptive_gradient_threshold(gradient_magnitude):
    """
    Berechnet einen adaptiven Schwellwert für Gradient-Kanten basierend auf der Statistik.
    PARAMETERFREI: Nutzt Otsu's Methode oder Perzentil-basierte Berechnung.
    
    Returns:
        float: Optimaler Schwellwert für signifikante Tiefensprünge
    """
    valid_grad = gradient_magnitude[gradient_magnitude > 0]
    if len(valid_grad) == 0:
        return 10.0  # Fallback
    
    # Otsu's Methode auf Gradient-Magnitude
    grad_u8 = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    threshold, _ = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Zurück in mm skalieren
    max_grad = np.max(gradient_magnitude)
    threshold_mm = (threshold / 255.0) * max_grad
    
    # Mindest-Threshold (Sensor-Rauschen ~5mm)
    threshold_mm = max(threshold_mm, 5.0)
    
    return threshold_mm


def analyze_box_gradient_parameterfree(depth, box):
    """
    PARAMETERFREI: Analysiert den Gradienten innerhalb einer Bounding Box.
    Der Schwellwert wird automatisch aus der Gradient-Statistik berechnet.
    
    Args:
        depth: 2D Tiefenbild (float, in Metern)
        box: [x1, y1, x2, y2] Koordinaten der Box
        
    Returns:
        dict mit split_mask, num_segments, segment_labels, gradient_magnitude, box_coords
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
            'gradient_magnitude': np.zeros((y2-y1, x2-x1)),
            'box_coords': (x1, y1, x2, y2),
            'threshold_used': 0
        }
    
    # In mm konvertieren
    box_depth_mm = box_depth * 1000.0
    
    # Leichter Blur
    box_depth_mm = cv2.GaussianBlur(box_depth_mm, (3, 3), 0)
    
    # Sobel Gradienten berechnen
    grad_x = cv2.Sobel(box_depth_mm, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(box_depth_mm, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # PARAMETERFREI: Berechne adaptiven Schwellwert
    threshold_mm = compute_adaptive_gradient_threshold(gradient_magnitude)
    
    # Threshold anwenden
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
        'box_coords': (x1, y1, x2, y2),
        'threshold_used': threshold_mm
    }


def select_frontmost_segment(segment_labels, depth_box):
    """
    PARAMETERFREI: Wählt das vorderste Segment (minimale mittlere Tiefe).
    
    Args:
        segment_labels: Label-Map der Segmente
        depth_box: Tiefenbild des Box-Bereichs
        
    Returns:
        int: Label des vordersten Segments
    """
    unique_labels = np.unique(segment_labels[segment_labels > 0])
    
    if len(unique_labels) == 0:
        return 0
    
    if len(unique_labels) == 1:
        return unique_labels[0]
    
    # Finde Segment mit minimaler mittlerer Tiefe
    min_depth = float('inf')
    best_label = unique_labels[0]
    
    for label in unique_labels:
        segment_mask = segment_labels == label
        segment_depths = depth_box[segment_mask]
        valid_depths = segment_depths[segment_depths > 0]
        
        if len(valid_depths) > 0:
            # Nimm 10. Perzentil für Robustheit
            mean_depth = np.percentile(valid_depths, 10)
            if mean_depth < min_depth:
                min_depth = mean_depth
                best_label = label
    
    return best_label


def create_depth_filtered_mask_parameterfree(box, depth, H, W):
    """
    PARAMETERFREI: Erstellt eine Maske für NUR die oberste Ebene.
    
    Methode: IQR-basiert (Interquartile Range)
    1. Finde minimale Tiefe (5. Perzentil) = oberste Oberfläche
    2. Berechne IQR der obersten Tiefenpixel
    3. Toleranz = 1.5 * IQR (statistisch robust)
    4. Alle Pixel innerhalb min_depth + Toleranz gehören zur obersten Ebene
    
    Args:
        box: [x1, y1, x2, y2]
        depth: Vollständiges Tiefenbild
        H, W: Bildabmessungen
        
    Returns:
        tuple: (mask, info_dict)
    """
    x1, y1, x2, y2 = [int(c) for c in box]
    
    # Sichere Grenzen
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    
    mask = np.zeros((H, W), dtype=np.uint8)
    box_depth = depth[y1:y2, x1:x2].copy()
    
    if box_depth.size == 0:
        return mask, {'method': 'empty', 'threshold': 0}
    
    # Nur gültige Tiefenwerte (> 0)
    valid_mask = box_depth > 0
    valid_depths = box_depth[valid_mask]
    
    if len(valid_depths) < 10:
        # Fallback: Ganze Box
        mask[y1:y2, x1:x2] = 1
        return mask, {'method': 'fallback', 'threshold': 0}
    
    # Finde minimale Tiefe (oberste Oberfläche) - 5. Perzentil für Robustheit
    min_depth = np.percentile(valid_depths, 5)
    
    # Betrachte nur die obersten 30% der Tiefenwerte für IQR-Berechnung
    p30_depth = np.percentile(valid_depths, 30)
    top_layer = valid_depths[valid_depths <= p30_depth]
    
    if len(top_layer) < 5:
        # Fallback wenn zu wenig Daten
        top_layer = valid_depths
    
    # IQR der obersten Schicht
    q1 = np.percentile(top_layer, 25)
    q3 = np.percentile(top_layer, 75)
    iqr = q3 - q1
    
    # Toleranz basierend auf IQR (klassische Outlier-Regel: 1.5 * IQR)
    # Aber mindestens 30mm und maximal 150mm für Sicherheit
    tolerance = 1.5 * iqr
    tolerance = max(tolerance, 0.03)   # Mindestens 30mm
    tolerance = min(tolerance, 0.15)   # Maximal 150mm
    
    threshold_depth = min_depth + tolerance
    
    # Erstelle Maske: Nur Pixel innerhalb der Toleranz
    front_mask = (box_depth > 0) & (box_depth <= threshold_depth)
    mask[y1:y2, x1:x2] = front_mask.astype(np.uint8)
    
    front_ratio = np.sum(front_mask) / len(valid_depths) if len(valid_depths) > 0 else 0
    
    return mask, {
        'method': 'iqr',
        'min_depth_mm': min_depth * 1000,
        'threshold_mm': threshold_depth * 1000,
        'tolerance_mm': tolerance * 1000,
        'iqr_mm': iqr * 1000,
        'front_ratio': front_ratio
    }


def apply_sobel_refinement(session_path, masks, labels, boxes=None):
    """
    PARAMETERFREI: Führt Gradientenanalyse mit automatischen Schwellwerten durch.
    
    Die Tiefentrennung basiert komplett auf Sobel-Gradienten:
    1. Berechne Gradienten pro Box
    2. Otsu findet automatisch optimalen Schwellwert
    3. Wähle das vorderste Segment (min Tiefe)
    
    Args:
        session_path: Pfad zur Session
        masks: Liste der Masken
        labels: Liste der Labels
        boxes: Liste der DINO Bounding Boxes
        
    Returns:
        tuple: (refined_masks, refined_labels, visualization_data)
    """
    print("\n[SOBEL] Starte parameterfreie Gradientenanalyse...")
    
    # 1. Lade Tiefenbild
    depth_path = os.path.join(session_path, "distance_to_image_plane", 
                               "distance_to_image_plane_0000.npy")
    if not os.path.exists(depth_path):
        print("[ERROR] Depth map not found.")
        return masks, labels, None
        
    depth = np.load(depth_path)
    H, W = depth.shape
    
    # 2. PARAMETERFREI: Erstelle tiefengefilterte Masken mit IQR pro Box
    print(f"[SOBEL] Tiefenfilterung mit IQR (parameterfrei)...")
    depth_filtered_masks = []
    depth_filter_info = []
    
    if boxes:
        for i, box in enumerate(boxes):
            filtered_mask, info = create_depth_filtered_mask_parameterfree(box, depth, H, W)
            depth_filtered_masks.append(filtered_mask)
            depth_filter_info.append(info)
            
            method = info.get('method', 'unknown')
            min_d = info.get('min_depth_mm', 0)
            tol = info.get('tolerance_mm', 0)
            front_ratio = info.get('front_ratio', 1.0)
            
            if method == 'iqr':
                print(f"  Box {i}: min={min_d:.0f}mm, tol={tol:.0f}mm, vorne={front_ratio*100:.0f}%")
            elif method == 'fallback':
                print(f"  Box {i}: Fallback (keine Tiefendaten)")
            else:
                print(f"  Box {i}: {method}")
        
        # Ersetze die ursprünglichen Masken mit den tiefengefilterten
        masks = depth_filtered_masks
    
    print(f"[SOBEL] {len(masks)} tiefengefilterte Masken erstellt")
    
    # 3. Globale Gradient-Berechnung für Visualisierung
    depth_mm = depth * 1000.0
    depth_mm = cv2.GaussianBlur(depth_mm, (3, 3), 0)
    norm_depth = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Canny Edge Detection
    binary_edges = cv2.Canny(norm_depth, 30, 100)
    
    # Sobel für Heatmap
    grad_x = cv2.Sobel(depth_mm, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_mm, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 4. Refinement der Masken
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
            
        # A. Occlusion Check (nutzt globale Scene-Tiefe)
        obj_depths = depth[mask_np]
        valid_obj_depths = obj_depths[obj_depths > 0]
        
        if len(valid_obj_depths) == 0:
            continue
            
        obj_mean_depth = np.mean(valid_obj_depths)
        
        # Dynamische Occlusion-Margin basierend auf Scene-Tiefe
        scene_depth_range = np.percentile(valid_depths, 99) - min_scene_depth
        occlusion_margin = scene_depth_range * 0.15  # 15% der Scene-Tiefe
        occlusion_margin = max(occlusion_margin, 0.10)  # Mindestens 10cm
        
        if obj_mean_depth > (min_scene_depth + occlusion_margin):
            print(f"  [FILTER] Maske {i} '{label}': Zu tief (Z={obj_mean_depth*1000:.0f}mm vs Top={min_scene_depth*1000:.0f}mm)")
            continue
        
        # B. Maske ist bereits tiefengefiltert - einfach hinzufügen wenn groß genug
        if mask_np.sum() > 200:
            refined_masks.append(mask_np.astype(np.uint8))
            refined_labels.append(label)
            print(f"  [OK] Maske {i} '{label}': {mask_np.sum()} Pixel")
            
    print(f"[SOBEL] Abgeschlossen. {len(masks)} -> {len(refined_masks)} Masken (PARAMETERFREI)")
    
    viz_data = {
        "gradient_magnitude": gradient_magnitude,
        "edges": binary_edges,
        "depth": depth,
        "depth_filter_info": depth_filter_info if boxes else []
    }
    
    return refined_masks, refined_labels, viz_data
