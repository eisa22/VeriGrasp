
import cv2
import numpy as np
import os
from PIL import Image

def apply_sobel_refinement(session_path, masks, labels):
    """
    Führt eine Gradientenanalyse mit Sobel auf dem Tiefenbild durch.
    Ziele:
    1. Erkennung von Spalten/Lücken (starke Tiefen-Sprünge).
    2. Präzisere Kanten.
    3. Filterung von verdeckten (occluded) Objekten.
    
    Args:
        session_path: Pfad zur Session (für Depth Image)
        masks: Liste der SAM-Masken
        labels: Liste der Labels
        
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
    
    # 2. Sobel-Berechnung (Kanten erkennen)
    # Skaliere Tiefe für bessere Gradienten-Berechnung (in mm)
    depth_mm = depth * 1000.0
    
    # Noise Reduction: Gaussian Blur um "Kriseln" zu entfernen -> Glattere Kanten
    depth_mm = cv2.GaussianBlur(depth_mm, (5, 5), 0)
    
    # Sobel X und Y
    sobel_x = cv2.Sobel(depth_mm, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_mm, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradienten-Magnitude (Kantenstärke)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalisiere für Visualisierung/Thresholding
    # Ein Gradient von > 10mm pro Pixel ist eine starke Kante (Spalte)
    EDGE_THRESHOLD = 15.0 
    edges = gradient_magnitude > EDGE_THRESHOLD
    
    # 3. Analyse & Refinement der Masken
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
        # Prüfe, ob das Objekt signifikant tiefer liegt als die oberste Ebene
        # "Occluded" = "liegt tiefer / hinter anderen Objekten"
        obj_depths = depth[mask_np]
        obj_mean_depth = np.mean(obj_depths)
        
        # Wenn Objekt > 20cm tiefer liegt als Top-Level -> Occluded/Background?
        # User sagt: "Objekte die occluded sind will ich nicht erkannt haben"
        OCCLUSION_MARGIN = 0.20 # 20cm Toleranz
        
        if obj_mean_depth > (min_scene_depth + OCCLUSION_MARGIN):
            print(f"  [FILTER] Maske {i} '{label}' entfernt: Zu tief/verdeckt (Z={obj_mean_depth*1000:.0f}mm vs Top={min_scene_depth*1000:.0f}mm)")
            continue
            
        # B. Kanten-Check & Reclaim (Ohne Watershed)
        # Strategie:
        # 1. Split: Maske MINUS Kanten -> Eindeutige "Kerne"
        # 2. Select: Wähle den Kern im Zentrum
        # 3. Reclaim: Füge die Kanten pixelweise wieder hinzu, die den Kern berühren
        
        internal_edges = edges & mask_np
        edge_pixel_count = np.sum(internal_edges)
        
        if edge_pixel_count > 0:
            print(f"  [REFINE] Maske {i} '{label}': {edge_pixel_count} Edge-Pixel gefunden. Optimiere Ränder...")
            
            # 1. Split: Ziehe Edge-Pixel ab
            core_mask = mask_np & (~edges)
            
            # Morphologie (Clean up noise)
            kernel = np.ones((3,3), np.uint8)
            core_mask_u8 = core_mask.astype(np.uint8)
            core_mask_u8 = cv2.morphologyEx(core_mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 2. Select: Finde Center Component
            num_labels, labels_im = cv2.connectedComponents(core_mask_u8)
            
            selected_core = None
            
            if num_labels > 2: # Mehr als 1 Objekt (plus Hintergrund)
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
                # 3. Hybrid Refinement: Watershed "hinten dran"
                # Wir nutzen den selektierten Kern (der sicher das richtige Objekt ist) als Seed für Watershed.
                # Damit wächst er genau bis zu den Kanten.
                
                print(f"    -> Starte Watershed-Wachstum vom Kern aus...")
                
                # Markers definieren
                markers = np.zeros_like(mask_np, dtype=np.int32)
                
                # Sure Foreground = Unser selektierter Kern
                # Leicht erodieren für Sicherheit? Nein, Kern ist schon sicher (edges wurden abgezogen).
                markers[selected_core == 1] = 2
                
                # Sure Background = Alles außerhalb der Box (STRIKT)
                # Wir erlauben kein Wachstum über die originale Box hinaus.
                # sure_bg ist alles, was NICHT in der originalen Box ist.
                sure_bg = 1 - mask_np.astype(np.uint8)
                markers[sure_bg == 1] = 1
                
                # Unknown = 0 (Automatisch, da np.zeros init)
                # Das ist jetzt genau der Bereich INNERHALB der Box, der NICHT 'sure_fg' ist.
                
                # Gradient für Watershed
                grad_u8 = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                grad_bgr = cv2.cvtColor(grad_u8, cv2.COLOR_GRAY2BGR)
                
                # Watershed
                cv2.watershed(grad_bgr, markers)
                
                # Ergebnis extrahieren (markers == 2)
                final_mask = (markers == 2).astype(np.uint8)
                
                # OPTIONAL: Rand-Pixel inkludieren (Boundary Inclusion)
                # Nur innerhalb der originalen Box!
                boundaries = (markers == -1).astype(np.uint8)
                dilated_obj = cv2.dilate(final_mask, np.ones((3,3), np.uint8), iterations=1)
                valid_boundaries = boundaries & dilated_obj
                
                # Final mask combinieren
                final_mask = final_mask | valid_boundaries
                
                # Sicherheits-Clip: Darf nicht größer als Original-Box sein
                final_mask = final_mask & mask_np.astype(np.uint8)
                
                refined_masks.append(final_mask)
                refined_labels.append(label)
            else:
                # Wenn Core ganz verschwunden ist (selten), nimm Original
                refined_masks.append(mask)
                refined_labels.append(label)
        else:
             refined_masks.append(mask)
             refined_labels.append(label)
            
    print(f"[SOBEL] Abgeschlossen. {len(masks)} -> {len(refined_masks)} Masken.")
    
    # Daten für Visualisierung
    viz_data = {
        "gradient_magnitude": gradient_magnitude,
        "edges": edges,
        "depth": depth
    }
    
    return refined_masks, refined_labels, viz_data
