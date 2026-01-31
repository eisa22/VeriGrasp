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
from config import DEBUG, DINO_MODEL_ID
import torch
import numpy as np
from transformers import (
    AutoProcessor as DinoProcessor,
    AutoModelForZeroShotObjectDetection as DinoModel
)


def convert_boxes_to_masks(boxes, height, width):
    """Konvertiert Bounding Boxes in binäre Masken."""
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
    boxes, scores, labels, orig_image = run_grounding_dino_only(session_path, dino_model, dino_processor)
    if not boxes:
        return
    
    # Phase 2: Box to Mask (SAM ersetzt)
    width, height = orig_image.size
    masks = convert_boxes_to_masks(boxes, height, width)
    
    # Scores und Labels bleiben gleich
    if not masks:
        return
    
    # Phase 3: Sobel Gradient Analysis & Refinement
    original_masks = [m.copy() for m in masks]
    original_labels = labels.copy()
    
    refined_masks, refined_labels, sobel_viz_data = apply_sobel_refinement(session_path, masks, labels)
    
    # Phase 5: Visualisierung
    results = None
    if DEBUG:
        results = visualize_3d(session_path, refined_masks, refined_labels, sobel_viz_data, original_masks, original_labels)
    
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
