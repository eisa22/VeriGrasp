
import cv2
import numpy as np
import os
from PIL import Image
from scipy import ndimage
from config import Z_ALIGN_MIN_KEEP_RATIO


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


def _compute_iqr_tolerance(depths_m):
    """IQR-Toleranz in Metern (min 30 mm, max 150 mm)."""
    if len(depths_m) < 5:
        return 0.03
    q1 = np.percentile(depths_m, 25)
    q3 = np.percentile(depths_m, 75)
    iqr = q3 - q1
    tol = 1.5 * iqr
    return float(max(tol, 0.03))


def estimate_z_plane_from_split_mask(depth, split_mask_local, box_coords):
    """
    Schätzt Z-Ebene und Toleranz aus Gradient-Kanten (Median + IQR).
    Returns:
        (z_plane_m, tolerance_m) oder (None, None)
    """
    x1, y1, x2, y2 = box_coords
    H, W = depth.shape
    edge_depths = []
    bh, bw = split_mask_local.shape
    for by in range(bh):
        for bx in range(bw):
            if split_mask_local[by, bx] > 0:
                gy, gx = y1 + by, x1 + bx
                if 0 <= gy < H and 0 <= gx < W and depth[gy, gx] > 0:
                    edge_depths.append(depth[gy, gx])
    if len(edge_depths) < 3:
        return None, None
    edge_depths = np.array(edge_depths)
    z_plane = float(np.median(edge_depths))
    tolerance = min(_compute_iqr_tolerance(edge_depths), 0.15)
    return z_plane, tolerance


def build_z_slab_mask(depth, box_coords, z_plane, tolerance):
    """HxW Maske: Pixel in der Box auf der Gradient-Z-Ebene."""
    x1, y1, x2, y2 = [int(c) for c in box_coords]
    H, W = depth.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    slab = np.zeros((H, W), dtype=np.uint8)
    box_d = depth[y1:y2, x1:x2]
    local = (box_d > 0) & (np.abs(box_d - z_plane) <= tolerance)
    slab[y1:y2, x1:x2] = local.astype(np.uint8)
    return slab


def analyze_box_gradient_parameterfree(depth, box, depth_constraint_mask=None, pallet_relative=False):
    """
    PARAMETERFREI: Analysiert den Gradienten innerhalb einer Bounding Box.
    Der Schwellwert wird automatisch aus der Gradient-Statistik berechnet.
    Gradienten werden nur auf der IQR-Vorderebene der Box berechnet.
    
    Args:
        depth: 2D Tiefenbild (float, in Metern)
        box: [x1, y1, x2, y2] Koordinaten der Box
        depth_constraint_mask: optional HxW – nur hier analysieren (Z-Slab)
        
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
            'threshold_used': 0,
            'front_layer_mask': np.zeros((max(y2 - y1, 0), max(x2 - x1, 0)), dtype=np.uint8),
        }
    
    # IQR-Vorderebene: Gradient nur auf oberster Schicht in der DINO-Box
    front_mask_full, _ = create_depth_filtered_mask_parameterfree(
        box, depth, H, W, pallet_relative=pallet_relative
    )
    front_local = front_mask_full[y1:y2, x1:x2]
    box_depth_masked = box_depth.copy()
    box_depth_masked[front_local == 0] = 0

    if depth_constraint_mask is not None:
        constraint_local = depth_constraint_mask[y1:y2, x1:x2]
        box_depth_masked[constraint_local == 0] = 0
    
    # In mm konvertieren (Hintergrund = 0 bleibt flach)
    box_depth_mm = box_depth_masked * 1000.0
    
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
        'threshold_used': threshold_mm,
        'front_layer_mask': front_local.astype(np.uint8),
    }


def analyze_box_gradient_z_aligned(depth, box, pallet_relative=False):
    """
    Zuerst grobe Gradienten in der DINO-Box, dann Z-Ebene der Kante schätzen,
    ROI auf Tiefen-Slab legen, danach Gradienten + Segmente nur in dieser Ebene.

    Returns:
        dict wie analyze_box_gradient_parameterfree, plus z_plane_m, z_tolerance_m,
        z_slab_mask (HxW), used_z_slab (bool)
    """
    coarse = analyze_box_gradient_parameterfree(depth, box, pallet_relative=pallet_relative)
    x1, y1, x2, y2 = coarse["box_coords"]
    split_coarse = coarse["split_mask"]

    z_plane, z_tol = estimate_z_plane_from_split_mask(
        depth, split_coarse, (x1, y1, x2, y2)
    )
    if z_plane is None:
        coarse["used_z_slab"] = False
        return coarse

    z_slab = build_z_slab_mask(depth, (x1, y1, x2, y2), z_plane, z_tol)
    front_local = coarse.get("front_layer_mask")
    if front_local is not None:
        front_full = np.zeros_like(z_slab)
        front_full[y1:y2, x1:x2] = front_local
        z_slab = ((z_slab > 0) & (front_full > 0)).astype(np.uint8)

    if int(z_slab.sum()) < 50:
        coarse["used_z_slab"] = False
        return coarse

    fine = analyze_box_gradient_parameterfree(
        depth, box, depth_constraint_mask=z_slab, pallet_relative=pallet_relative
    )
    fine["z_plane_m"] = z_plane
    fine["z_plane_mm"] = z_plane * 1000
    fine["z_tolerance_m"] = z_tol
    fine["z_tolerance_mm"] = z_tol * 1000
    fine["z_slab_mask"] = z_slab
    fine["used_z_slab"] = True
    return fine


def align_mask_to_depth_plane(global_mask, depth, split_mask_local, box_coords,
                              min_keep_ratio=None, front_layer_mask=None):
    """
    Schneidet eine 2D-Segmentmaske auf die Z-Ebene der Gradient-Kante.
    Die Gradient-Kante definiert die Referenz-Z-Ebene; ein großer z_residual
    vor dem Schnitt ist kein Reject-Grund (DINO ist rein 2D).

    Args:
        global_mask: HxW uint8 Maske
        depth: HxW Tiefenbild in Metern
        split_mask_local: Kantenmaske im lokalen Box-Koordinatensystem
        box_coords: (x1, y1, x2, y2)
        min_keep_ratio: Min. Anteil erhaltener Pixel nach Z-Schnitt
        front_layer_mask: optional HxW, Schnitt mit Vorderebene vor Z-Align

    Returns:
        tuple: (aligned_mask, z_stats) – aligned_mask ist None bei Reject
    """
    if min_keep_ratio is None:
        min_keep_ratio = Z_ALIGN_MIN_KEEP_RATIO

    x1, y1, x2, y2 = box_coords
    H, W = depth.shape
    work_mask = global_mask.copy()

    if front_layer_mask is not None:
        work_mask = (work_mask > 0) & (front_layer_mask > 0)
        work_mask = work_mask.astype(np.uint8)

    # Tiefe der Gradient-Kante (global) – maßgeblich für Z-Ebene
    edge_depths_list = []
    box_h, box_w = split_mask_local.shape
    for by in range(box_h):
        for bx in range(box_w):
            if split_mask_local[by, bx] > 0:
                gy, gx = y1 + by, x1 + bx
                if 0 <= gy < H and 0 <= gx < W and depth[gy, gx] > 0:
                    edge_depths_list.append(depth[gy, gx])

    seg_ys, seg_xs = np.where(work_mask > 0)
    if len(seg_ys) == 0:
        return None, {'reject_reason': 'empty_segment'}

    seg_depths = depth[seg_ys, seg_xs]
    seg_depths = seg_depths[seg_depths > 0]
    if len(seg_depths) == 0:
        return None, {'reject_reason': 'no_valid_depth'}

    if edge_depths_list:
        edge_depths_arr = np.array(edge_depths_list)
        z_edge = float(np.median(edge_depths_arr))
        tolerance = _compute_iqr_tolerance(edge_depths_arr)
    else:
        z_edge = float(np.median(seg_depths))
        tolerance = _compute_iqr_tolerance(seg_depths)

    z_interior_median = float(np.median(seg_depths))
    z_residual = abs(z_interior_median - z_edge)

    # Z-Schnitt auf Gradient-Ebene (behebt DINO↔Gradient Z-Verdrehung)
    depth_ok = (depth > 0) & (np.abs(depth - z_edge) <= tolerance)
    aligned = (work_mask > 0) & depth_ok
    aligned = aligned.astype(np.uint8)

    original_count = int(global_mask.sum())
    kept_count = int(aligned.sum())
    pixels_kept_ratio = kept_count / original_count if original_count > 0 else 0.0

    kept_depths = depth[aligned > 0]
    kept_depths = kept_depths[kept_depths > 0]
    kept_spread = float(np.percentile(kept_depths, 75) - np.percentile(kept_depths, 25)) if len(kept_depths) > 5 else 0.0

    z_stats = {
        'z_plane_m': z_edge,
        'z_plane_mm': z_edge * 1000,
        'tolerance_m': tolerance,
        'tolerance_mm': tolerance * 1000,
        'z_residual_m': z_residual,
        'z_residual_mm': z_residual * 1000,
        'pixels_kept_ratio': pixels_kept_ratio,
        'original_pixels': original_count,
        'kept_pixels': kept_count,
        'kept_depth_iqr_mm': kept_spread * 1000,
    }

    if kept_count < 50:
        z_stats['reject_reason'] = 'too_few_pixels'
        return None, z_stats

    if pixels_kept_ratio < min_keep_ratio:
        z_stats['reject_reason'] = 'low_keep_ratio'
        return None, z_stats

    # Nur ablehnen wenn das Ergebnis nach Schnitt selbst noch stark gestreut ist
    if kept_spread > 3.0 * tolerance:
        z_stats['reject_reason'] = 'kept_depth_scatter'
        return None, z_stats

    return aligned, z_stats


def select_frontmost_segment(segment_labels, depth_box, pallet_relative=False):
    """
    PARAMETERFREI: Wählt das vorderste Segment.
    Absolut: minimale Tiefe; pallet_relative: maximale Höhe über Palette.
    
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
    
    best_depth = float('-inf') if pallet_relative else float('inf')
    best_label = unique_labels[0]
    
    for label in unique_labels:
        segment_mask = segment_labels == label
        segment_depths = depth_box[segment_mask]
        valid_depths = segment_depths[segment_depths > 0]
        
        if len(valid_depths) > 0:
            pct = 90 if pallet_relative else 10
            mean_depth = np.percentile(valid_depths, pct)
            if pallet_relative:
                if mean_depth > best_depth:
                    best_depth = mean_depth
                    best_label = label
            elif mean_depth < best_depth:
                best_depth = mean_depth
                best_label = label
    
    return best_label


def create_depth_filtered_mask_parameterfree(box, depth, H, W, pallet_relative=False):
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
    
    if pallet_relative:
        ref_depth = np.percentile(valid_depths, 95)
        p70_depth = np.percentile(valid_depths, 70)
        top_layer = valid_depths[valid_depths >= p70_depth]
    else:
        ref_depth = np.percentile(valid_depths, 5)
        p30_depth = np.percentile(valid_depths, 30)
        top_layer = valid_depths[valid_depths <= p30_depth]
    
    if len(top_layer) < 5:
        top_layer = valid_depths
    
    q1 = np.percentile(top_layer, 25)
    q3 = np.percentile(top_layer, 75)
    iqr = q3 - q1
    
    tolerance = 1.5 * iqr
    tolerance = max(tolerance, 0.03)
    tolerance = min(tolerance, 0.15)
    
    if pallet_relative:
        threshold_depth = ref_depth - tolerance
        front_mask = (box_depth > 0) & (box_depth >= threshold_depth)
    else:
        threshold_depth = ref_depth + tolerance
        front_mask = (box_depth > 0) & (box_depth <= threshold_depth)
    
    # Morphologische Erosion um unsaubere Ränder zu bereinigen
    # Das verhindert Überlappungen mit benachbarten Objekten
    kernel = np.ones((3, 3), np.uint8)
    front_mask_clean = cv2.erode(front_mask.astype(np.uint8), kernel, iterations=1)
    
    # Wenn Erosion zu viel entfernt hat, nimm Original
    if front_mask_clean.sum() < front_mask.sum() * 0.5:
        front_mask_clean = front_mask.astype(np.uint8)
    
    mask[y1:y2, x1:x2] = front_mask_clean
    
    front_ratio = np.sum(front_mask_clean) / len(valid_depths) if len(valid_depths) > 0 else 0
    
    return mask, {
        'method': 'iqr',
        'min_depth_mm': ref_depth * 1000,
        'threshold_mm': threshold_depth * 1000,
        'tolerance_mm': tolerance * 1000,
        'iqr_mm': iqr * 1000,
        'front_ratio': front_ratio
    }


def apply_sobel_refinement(session_path, masks, labels, boxes=None, session_context=None):
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

    pallet_relative = session_context is not None
    if session_context is not None:
        from Segmentation.pallet_scene import get_working_depth
        depth = get_working_depth(session_context)
    else:
        depth_path = os.path.join(
            session_path,
            "distance_to_image_plane",
            "distance_to_image_plane_0000.npy",
        )
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
            filtered_mask, info = create_depth_filtered_mask_parameterfree(
                box, depth, H, W, pallet_relative=pallet_relative
            )
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
            
        # Maske ist bereits tiefengefiltert - einfach hinzufügen wenn groß genug
        if mask_np.sum() > 200:
            refined_masks.append(mask_np.astype(np.uint8))
            refined_labels.append(label)
            print(f"  [OK] Maske {i} '{label}': {mask_np.sum()} Pixel")
            
    print(f"[SOBEL] Abgeschlossen. {len(masks)} -> {len(refined_masks)} Masken (PARAMETERFREI)")
    
    viz_data = {
        "gradient_magnitude": gradient_magnitude,
        "edges": binary_edges,
        "depth": depth,
        "depth_filter_info": depth_filter_info if boxes else [],
        "session_context": session_context,
    }
    
    return refined_masks, refined_labels, viz_data
