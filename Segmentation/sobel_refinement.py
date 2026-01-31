
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
            
        # B. Kanten-Check
        # Prüfe, ob starke Kanten DURCH das Objekt laufen (Spaltung nötig?)
        # Dies ist komplex, hier vereinfacht:
        # Wir nutzen die Sobel-Edges um die Maske "einzuschränken", falls sie über Kanten "blutet".
        # Idee: Maske AND (NOT Edges) -> Maske wird an Kanten unterbrochen
        
        # Erstelle Edge-Maske innerhalb des Objekts
        internal_edges = edges & mask_np
        edge_pixel_count = np.sum(internal_edges)
        
        if edge_pixel_count > 0:
            print(f"  [REFINE] Maske {i} '{label}': {edge_pixel_count} Edge-Pixel gefunden. Optimiere Ränder...")
            # Einfache Refinement-Strategie: Ziehe Edge-Pixel von der Maske ab
            # Das trennt 'blutende' Masken an starken Tiefen-Kanten
            refined_mask = mask_np & (~edges)
            
            # Morphologie um Rauschen zu entfernen (Closing/Opening)
            kernel = np.ones((3,3), np.uint8)
            refined_mask_u8 = refined_mask.astype(np.uint8)
            refined_mask_u8 = cv2.morphologyEx(refined_mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Prüfen ob Maske zerfallen ist (durch Kanten-Schnitt) -> Behalte größtes Stück?
            # Oder behalte alle Stücke als separates Objekte?
            # User: "Spalten erkennen". Wenn Sobel die Maske teilt, sollten wir das nutzen.
            num_labels, labels_im = cv2.connectedComponents(refined_mask_u8)
            
            if num_labels > 2: # 1 is background, so > 1 object found
                 print(f"    -> Maske wurde durch Kanten in {num_labels-1} Teile gespalten!")
                 # Füge alle Teile als separate Masken hinzu
                 for comp_idx in range(1, num_labels):
                     comp_mask = (labels_im == comp_idx).astype(np.uint8)
                     if comp_mask.sum() > 200: # Min Size
                         refined_masks.append(comp_mask)
                         refined_labels.append(f"{label}_part{comp_idx}")
            else:
                if refined_mask_u8.sum() > 200:
                    refined_masks.append(refined_mask_u8)
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
