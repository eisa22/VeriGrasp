"""
main.py
Hauptpipeline für Pallet-Segmentierung.

Pipeline: DINO → SAM → SAM3D → Deduplizierung → Visualisierung → Screenshots → LLM
"""
from GroundingSAM.grounding_sam import run_grounding_dino_only, generate_sam_masks_for_boxes
from Sam3D.sam3d import refine_masks_3d, deduplicate_masks_3d
from Visualization.visualizer import visualize_3d, capture_scene_screenshots
from LLMOrchestrator.orchestrator import run_orchestrator
from path_utils import get_all_session_paths
from config import DEBUG, SAM_MODEL_ID
import torch
from transformers import (
    SamProcessor, 
    SamModel,
    AutoProcessor as DinoProcessor,
    AutoModelForZeroShotObjectDetection as DinoModel
)
from config import DEBUG, SAM_MODEL_ID, DINO_MODEL_ID


def process_session(session_path, dino_model=None, dino_processor=None, sam_model=None, sam_processor=None):
    """Verarbeitet eine Session durch die gesamte Pipeline."""
    # Phase 1: DINO
    boxes, scores, labels, _ = run_grounding_dino_only(session_path, dino_model, dino_processor)
    if not boxes:
        return
    
    # Phase 2: SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Modelle sollten bereits geladen sein, Fallback falls nicht:
    if sam_model is None or sam_processor is None:
        print(f"Lade SAM Modell ({SAM_MODEL_ID}) auf {device}... (Dies kann beim ersten Mal einige Minuten dauern)")
        sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
        sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    masks, boxes, scores, labels = generate_sam_masks_for_boxes(
        session_path, boxes, labels, sam_model, sam_processor
    )
    if not masks:
        return
    
    # Phase 3: SAM3D
    masks, boxes, scores, labels = refine_masks_3d(masks, boxes, scores, labels, session_path)
    if not masks:
        return
    
    # Phase 4: Deduplizierung
    masks, boxes, scores, labels = deduplicate_masks_3d(masks, boxes, scores, labels, session_path)
    
    # Phase 5: Visualisierung
    results = None
    if DEBUG:
        results = visualize_3d(session_path, masks, labels)
    
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
    
    # 1. SAM Laden
    print(f"Lade SAM Modell ({SAM_MODEL_ID})...")
    sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    
    # 2. DINO Laden
    print(f"Lade DINO Modell ({DINO_MODEL_ID})...")
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)
    
    sessions = get_all_session_paths()
    for session_path in sessions:
        process_session(session_path, dino_model, dino_processor, sam_model, sam_processor)


if __name__ == "__main__":
    main()
