"""
main.py
Hauptpipeline für Pallet-Segmentierung.

Pipeline: DINO → SAM → SAM3D → Deduplizierung → Visualisierung → Screenshots → LLM
"""
from GroundingSAM.grounding_sam import run_grounding_dino_only
from Segmentation.sobel_refinement import apply_sobel_refinement
from Visualization.visualizer import visualize_3d, capture_scene_screenshots
from LLMOrchestrator.orchestrator import run_orchestrator
from path_utils import get_all_session_paths
from config import DEBUG, DINO_MODEL_ID, DEPTH_BASE_TOLERANCE_MM, DEPTH_MAX_TOLERANCE_MM, DEPTH_ADAPTIVE
import torch
import numpy as np
import os
from transformers import (
    AutoProcessor as DinoProcessor,
    AutoModelForZeroShotObjectDetection as DinoModel
)


def convert_boxes_to_depth_filtered_masks(boxes, height, width, depth, base_tolerance_mm=DEPTH_BASE_TOLERANCE_MM, max_tolerance_mm=DEPTH_MAX_TOLERANCE_MM, adaptive=DEPTH_ADAPTIVE):
    """
    Konvertiert Bounding Boxes in binäre Masken unter Berücksichtigung der Tiefe.
    
    Für jede Box wird nur die OBERSTE Ebene (minimale Tiefe) genommen.
    Pixel die deutlich tiefer liegen (andere gestapelte Pakete) werden NICHT zur Maske hinzugefügt.
    
    ADAPTIV: Bei gewölbten Objekten wie Säcken wird die Toleranz automatisch erhöht,
    basierend auf der Standardabweichung der Tiefenwerte innerhalb der obersten Schicht.
    
    Args:
        boxes: Liste von Bounding Boxes [x0, y0, x1, y1]
        height, width: Bildabmessungen
        depth: 2D Tiefenbild (in Metern)
        base_tolerance_mm: Basis-Toleranz in mm (für flache Objekte)
        adaptive: Wenn True, wird Toleranz basierend auf Oberflächen-Varianz angepasst
    
    Returns:
        Liste von gefilterten binären Masken
    """
    masks = []
    base_tolerance_m = base_tolerance_mm / 1000.0
    
    for box_idx, box in enumerate(boxes):
        mask = np.zeros((height, width), dtype=np.uint8)
        x0, y0, x1, y1 = [int(b) for b in box]
        
        # Clip to image boundaries
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width, x1)
        y1 = min(height, y1)
        
        # Extrahiere Tiefenwerte innerhalb der Box
        box_depth = depth[y0:y1, x0:x1].copy()
        
        if box_depth.size == 0:
            masks.append(mask)
            continue
        
        # Finde die minimale Tiefe (oberstes Objekt) innerhalb der Box
        valid_mask = box_depth > 0
        valid_depths = box_depth[valid_mask]
        
        if len(valid_depths) == 0:
            # Fallback: Alle Pixel nehmen
            mask[y0:y1, x0:x1] = 1
            masks.append(mask)
            continue
        
        # Nimm das 5. Perzentil als "oberste Ebene" (robuster gegen Rauschen)
        min_depth = np.percentile(valid_depths, 5)
        
        # ADAPTIVE TOLERANZ
        # Bei gewölbten Objekten (Säcke, unregelmäßige Formen) ist die Varianz höher
        if adaptive:
            # Betrachte nur die obersten 30% der Pixel (um Hintergrund auszuschließen)
            p30_depth = np.percentile(valid_depths, 30)
            top_layer_depths = valid_depths[valid_depths <= p30_depth]
            
            if len(top_layer_depths) > 10:
                # Standardabweichung der obersten Schicht
                std_depth = np.std(top_layer_depths)
                
                # Adaptive Toleranz: Basis + 3 * Standardabweichung
                # Bei flachen Paketen: std ~ 5-10mm -> Toleranz bleibt ~40-70mm
                # Bei gewölbten Säcken: std ~ 30-50mm -> Toleranz wird ~130-190mm
                adaptive_tolerance = base_tolerance_m + 3.0 * std_depth
                
                # Limit: Maximum aus Config (um zu verhindern dass darunterliegende Objekte eingeschlossen werden)
                max_tolerance_m = max_tolerance_mm / 1000.0
                adaptive_tolerance = min(adaptive_tolerance, max_tolerance_m)
                
                tolerance_m = adaptive_tolerance
            else:
                tolerance_m = base_tolerance_m
        else:
            tolerance_m = base_tolerance_m
        
        # Erstelle die gefilterte Maske (vektorisiert für Performance)
        full_box_mask = (depth[y0:y1, x0:x1] > 0) & (depth[y0:y1, x0:x1] <= (min_depth + tolerance_m))
        mask[y0:y1, x0:x1] = full_box_mask.astype(np.uint8)
        
        # Debug-Ausgabe
        pixel_count = mask.sum()
        if adaptive and 'std_depth' in dir():
            print(f"  [DEPTH-FILTER] Box {box_idx}: min_z={min_depth*1000:.0f}mm, std={std_depth*1000:.1f}mm, tolerance={tolerance_m*1000:.0f}mm, pixels={pixel_count}")
        
        masks.append(mask)
    
    return masks


def convert_boxes_to_masks(boxes, height, width):
    """Konvertiert Bounding Boxes in binäre Masken (OHNE Tiefenfilterung - Legacy)."""
    masks = []
    for box in boxes:
        mask = np.zeros((height, width), dtype=np.uint8)
        x0, y0, x1, y1 = [int(b) for b in box]
        # Clip to image boundaries
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width, x1)
        y1 = min(height, y1)
        mask[y0:y1, x0:x1] = 1
        masks.append(mask)
    return masks


def process_session(session_path, dino_model=None, dino_processor=None):
    """Verarbeitet eine Session durch die gesamte Pipeline."""
    # Phase 1: DINO
    boxes, scores, labels, orig_image, dino_debug = run_grounding_dino_only(session_path, dino_model, dino_processor)
    if not boxes:
        return
    
    # Lade Tiefenbild für tiefengefilterte Masken
    depth_path = os.path.join(session_path, "distance_to_image_plane", 
                               "distance_to_image_plane_0000.npy")
    depth = np.load(depth_path)
    
    # Phase 2: Box to Mask MIT Tiefenfilterung
    # Nur Pixel die zur obersten Ebene in jeder Box gehören werden genommen
    # Parameter kommen aus config.py (DEPTH_BASE_TOLERANCE_MM, DEPTH_MAX_TOLERANCE_MM, DEPTH_ADAPTIVE)
    width, height = orig_image.size
    masks = convert_boxes_to_depth_filtered_masks(boxes, height, width, depth)
    
    print(f"[MASK] {len(masks)} tiefengefilterte Masken erstellt (Basis: {DEPTH_BASE_TOLERANCE_MM}mm, Max: {DEPTH_MAX_TOLERANCE_MM}mm)")
    
    # Scores und Labels bleiben gleich
    if not masks:
        return
    
    # Phase 3: Sobel Gradient Analysis & Refinement
    original_masks = [m.copy() for m in masks]
    original_labels = labels.copy()
    
    refined_masks, refined_labels, sobel_viz_data = apply_sobel_refinement(session_path, masks, labels, boxes)
    
    # Phase 5: Visualisierung
    results = None
    if DEBUG:
        results = visualize_3d(
            session_path, 
            refined_masks, 
            refined_labels, 
            sobel_viz_data, 
            original_masks, 
            original_labels,
            dino_debug  # NEU: Debug-Daten für Box-Visualisierung
        )
    
    # Phase 6: Screenshots aufnehmen
    # screenshot_paths = capture_scene_screenshots(session_path, masks, labels)
    screenshot_paths = []
    
    # Phase 7: LLM Orchestrator - Paket-Auswahl
    # llm_result = run_orchestrator(screenshot_paths)
    llm_result = None
    
    return {
        "visualization": results, 
        #"screenshots": screenshot_paths,
        #"llm_decision": llm_result
    }


def main():
    """Hauptfunktion: Orchestriert die Pipeline für alle Sessions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initialisiere Pipeline auf: {device}")
    
    # 1. DINO Laden (SAM wird nicht mehr geladen)
    print(f"Lade DINO Modell ({DINO_MODEL_ID})...")
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)
    
    sessions = get_all_session_paths()
    for session_path in sessions:
        process_session(session_path, dino_model, dino_processor)


if __name__ == "__main__":
    main()
