"""
main.py
Hauptpipeline für Pallet-Segmentierung mit DINO + SAM.

Pipeline-Phasen:
1. Initiale Erkennung (Grounding DINO + SAM)
2. Filterung (Duplikate/Überlappungen entfernen)
3. Refinement (Depth-basierte Aufteilung großer Objekte)
4. Visualisierung (2D + 3D)
"""
import numpy as np
import os

# Import externe Module
from GroundingSAM.grounding_sam import run_grounding_sam
from path_utils import get_all_session_paths
from config import DEBUG

# Import eigene Module
from Segmentation.filtering import filter_overlapping_masks
from Segmentation.refinement import split_mask_by_depth_gaps
from Visualization.visualizer import visualize_3d


def process_session(session_path):
    """
    Verarbeitet eine einzelne Session.
    
    Args:
        session_path: Pfad zur Session (enthält rgb/ und distance_to_image_plane/)
        
    Returns:
        tuple: (boxes, masks, labels, scores) - finale Segmentierungsergebnisse
    """
    session_name = os.path.basename(session_path)
    print(f"\n{'='*60}")
    print(f"[SESSION] {session_name}")
    print(f"{'='*60}")
    
    # -------------------------------------------------------------------------
    # Phase 1: Initiale Erkennung mit Grounding DINO + SAM
    # -------------------------------------------------------------------------
    boxes, masks, scores, labels = run_grounding_sam(session_path)
    
    if len(masks) == 0:
        print(f"[SESSION] Keine Masken gefunden → überspringe Session.")
        return None
    
    # -------------------------------------------------------------------------
    # Phase 2: Filterung (Duplikate und große Überlappungen)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 2: FILTERUNG")
    print(f"{'='*60}")
    
    original_count = len(masks)
    boxes, masks, scores, labels = filter_overlapping_masks(boxes, masks, scores, labels)
    filtered_count = original_count - len(masks)
    print(f"\n→ {original_count} → {len(masks)} Masken (gefiltert: {filtered_count})")
    
    # -------------------------------------------------------------------------
    # Phase 3: Refinement großer Masken mit Depth-Gap-Detection
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 3: REFINEMENT (Depth-Gap Detection)")
    print(f"{'='*60}")
    
    # Lade Depth Map
    depth_path = os.path.join(session_path, "distance_to_image_plane",
                               "distance_to_image_plane_0000.npy")
    depth_map = np.load(depth_path)
    
    boxes, masks, scores, labels = refine_large_masks(
        boxes, masks, scores, labels, depth_map
    )
    
    print(f"\n[SESSION] {session_name}: {len(masks)} finale Objekte")
    
    # Visualisierung direkt nach der Session (wenn DEBUG=True)
    if DEBUG:
        print(f"\n{'='*60}")
        print(f"3D-VISUALISIERUNG: {session_name}")
        print(f"{'='*60}")
        visualize_3d(session_path, masks, labels)
    
    return boxes, masks, scores, labels, session_path


def refine_large_masks(boxes, masks, scores, labels, depth_map):
    """
    Teilt große Masken (> 15% Bildgröße) anhand von Tiefengradienten.
    
    Args:
        boxes, masks, scores, labels: Aktuelle Segmentierungsdaten
        depth_map: Tiefenkarte
        
    Returns:
        tuple: (neue_boxes, neue_masks, neue_scores, neue_labels)
    """
    total_pixels = masks[0].shape[0] * masks[0].shape[1]
    size_threshold = 0.15 * total_pixels
    
    new_boxes = []
    new_masks = []
    new_scores = []
    new_labels = []
    
    for i, (box, mask, score, label) in enumerate(zip(boxes, masks, scores, labels)):
        mask_size = np.sum(mask)
        
        # Nur große Masken splitten
        if mask_size > size_threshold:
            print(f"\nMaske {i} '{label}': {mask_size} Pixel → versuche Split")
            
            split_masks = split_mask_by_depth_gaps(mask, depth_map, min_segment_size=500)
            
            if len(split_masks) > 1:
                print(f"  → Aufgeteilt in {len(split_masks)} Segmente!")
                
                # Für jedes neue Segment: Box berechnen, Daten hinzufügen
                for j, split_mask in enumerate(split_masks):
                    ys, xs = np.where(split_mask > 0)
                    if len(xs) > 0:
                        new_box = [float(xs.min()), float(ys.min()), 
                                   float(xs.max()), float(ys.max())]
                        new_boxes.append(new_box)
                        new_masks.append(split_mask)
                        new_scores.append(score)
                        new_labels.append(f"{label}_{j+1}")
            else:
                # Keine Aufteilung möglich/nötig
                new_boxes.append(box)
                new_masks.append(mask)
                new_scores.append(score)
                new_labels.append(label)
        else:
            # Kleine Maske: unverändert übernehmen
            new_boxes.append(box)
            new_masks.append(mask)
            new_scores.append(score)
            new_labels.append(label)
    
    print(f"\n→ Nach Refinement: {len(new_masks)} Masken")
    return new_boxes, new_masks, new_scores, new_labels


def main():
    """Hauptfunktion: Orchestriert die gesamte Pipeline."""
    print(f"{'='*60}")
    print("PALLET SEGMENTATION PIPELINE")
    print(f"{'='*60}")
    
    # Alle Sessions laden
    all_sessions = get_all_session_paths()
    print(f"\n[MAIN] Gefundene Sessions: {len(all_sessions)}")
    
    results = []
    
    # Verarbeite jede Session
    for session_path in all_sessions:
        result = process_session(session_path)
        if result is not None:
            results.append(result)
    
    # Abschluss
    print(f"\n{'='*60}")
    print(f"[MAIN] Pipeline abgeschlossen: {len(results)}/{len(all_sessions)} Sessions erfolgreich")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
