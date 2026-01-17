"""
main.py
Hauptpipeline für Pallet-Segmentierung mit Hybrid DINO + SAM Grid-Prompts + SAM3D.

Pipeline-Phasen:
1. Grounding DINO - Grobe Regionen-Erkennung (mit Box-Filterung & NMS)
2. SAM Grid-Prompts - Präzise Segmentierung innerhalb jeder ROI
3. SAM3D - Selektives 3D-Splitting (nur große/mehrschichtige Masken)
4. 3D-Visualisierung mit Farben
"""
import os
import torch
from transformers import SamProcessor, SamModel

# Import externe Module
from GroundingSAM.grounding_sam import run_grounding_dino_only
from GroundingSAM.sam_grid_generator import generate_sam_masks_for_all_rois
from path_utils import get_all_session_paths
from config import DEBUG, SAM_MODEL_ID

# Import eigene Module
from Sam3D.sam3d import refine_masks_3d
from Visualization.visualizer import visualize_3d


def process_session(session_path):
    """
    Verarbeitet eine einzelne Session mit Hybrid-Pipeline.
    
    Args:
        session_path: Pfad zur Session (enthält rgb/ und distance_to_image_plane/)
    """
    session_name = os.path.basename(session_path)
    print(f"\n{'='*60}")
    print(f"[SESSION] {session_name}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # -------------------------------------------------------------------------
    # Phase 1: Grounding DINO - Grobe Regionen
    # -------------------------------------------------------------------------
    print(f"\n[PHASE 1] Grounding DINO - Regionen-Erkennung")
    boxes, scores, labels, orig_image = run_grounding_dino_only(session_path)
    
    if len(boxes) == 0:
        print(f"[SESSION] Keine Regionen gefunden → überspringe Session.")
        return
    
    print(f"→ {len(boxes)} ROIs von DINO (nach Filterung)")
    
    # -------------------------------------------------------------------------
    # Phase 2: SAM Grid-Prompts in ROIs
    # -------------------------------------------------------------------------
    print(f"\n[PHASE 2] SAM Grid-Prompts - Segmentierung in ROIs")
    
    # Lade SAM-Modelle einmal
    sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    
    masks, boxes, scores, labels = generate_sam_masks_for_all_rois(
        session_path, boxes, labels, sam_model, sam_processor
    )
    
    if len(masks) == 0:
        print(f"[SESSION] Keine Masken von SAM Grid → überspringe Session.")
        return
    
    print(f"→ {len(masks)} Masken von SAM Grid-Prompts")
    
    # -------------------------------------------------------------------------
    # Phase 3: SAM3D - Selektives 3D-Splitting
    # -------------------------------------------------------------------------
    print(f"\n[PHASE 3] SAM3D (Selektives 3D-Splitting)")
    masks, boxes, scores, labels = refine_masks_3d(
        masks, boxes, scores, labels, session_path
    )
    
    if len(masks) == 0:
        print(f"[SESSION] Keine Masken nach SAM3D → überspringe Session.")
        return
    
    print(f"→ {len(masks)} finale Objekte nach SAM3D")
    
    # -------------------------------------------------------------------------
    # Phase 4: 3D-Visualisierung mit Farben
    # -------------------------------------------------------------------------
    if DEBUG:
        print(f"\n[PHASE 4] 3D-Visualisierung")
        visualize_3d(session_path, masks, labels)


def main():
    """Hauptfunktion: Orchestriert die gesamte Pipeline."""
    print(f"{'='*60}")
    print("PALLET SEGMENTATION PIPELINE (HYBRID)")
    print("DINO (Regionen) → SAM Grid-Prompts → SAM3D (Selektiv) → Visualisierung")
    print(f"{'='*60}")
    
    # Alle Sessions laden
    all_sessions = get_all_session_paths()
    print(f"\n[MAIN] Gefundene Sessions: {len(all_sessions)}")

    # Verarbeite jede Session
    for session_path in all_sessions:
        process_session(session_path)

    # Abschluss
    print(f"\n{'='*60}")
    print(f"[MAIN] Pipeline abgeschlossen!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
