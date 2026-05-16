"""
main.py
Hauptpipeline für Pallet-Segmentierung.

Pipeline: DINO → Box-Masken → Sobel (parameterfrei) → Visualisierung (+ SAM3D parallel)
"""
from GroundingSAM.grounding_sam import run_grounding_dino_only
from Segmentation.pallet_scene import prepare_session_context
from Segmentation.sobel_refinement import apply_sobel_refinement
from Sam3D.sam3d import refine_masks_3d
from Visualization.visualizer import (
    visualize_3d,
    capture_scene_screenshots,
    extract_dino_gradient_masks,
)
from LLMOrchestrator.orchestrator import run_orchestrator
from path_utils import get_all_session_paths
from config import DEBUG, DINO_MODEL_ID, MATCH_CLOSURE_RATIO, MATCH_BORDER_TOUCH_RATIO
from perception.configs.load import load_bottom_inference_config
from perception.adapter import (
    build_candidates_from_closed_matches,
    build_candidates_from_sam3d,
    build_scene_pcd_from_depth,
)
from perception.bottom_inference import infer_bottom_planes
import torch
import numpy as np
import os
from transformers import (
    AutoProcessor as DinoProcessor,
    AutoModelForZeroShotObjectDetection as DinoModel
)


def convert_boxes_to_masks(boxes, height, width):
    """
    Konvertiert Bounding Boxes in binäre Masken.
    Die Tiefenfilterung erfolgt PARAMETERFREI in sobel_refinement.py.
    """
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
    # Phase 0: Palettenebene (z=0) + Workspace
    session_context = prepare_session_context(session_path)
    if session_context is None:
        print(f"[SKIP] Session ohne Depth: {session_path}")
        return

    # Phase 1: DINO
    boxes, scores, labels, orig_image, dino_debug = run_grounding_dino_only(
        session_path, dino_model, dino_processor, session_context=session_context
    )
    if not boxes:
        return
    
    # Phase 2: Box to Mask (einfach - Tiefenfilterung erfolgt in Phase 3)
    width, height = orig_image.size
    masks = convert_boxes_to_masks(boxes, height, width)
    
    print(f"[MASK] {len(masks)} Box-Masken erstellt")
    
    if not masks:
        return
    
    # Phase 3: PARAMETERFREI - Sobel Gradient Analysis & Tiefenfilterung
    # Die Tiefentrennung erfolgt automatisch basierend auf Gradienten
    original_masks = [m.copy() for m in masks]
    original_labels = labels.copy()
    
    refined_masks, refined_labels, sobel_viz_data = apply_sobel_refinement(
        session_path, masks, labels, boxes, session_context=session_context
    )

    # Phase 3b: DINO ∩ durchgängige Gradient-Kante → geschlossene Paket-Masken
    closed_matches = extract_dino_gradient_masks(
        session_path,
        dino_debug,
        sobel_viz_data,
        closure_ratio=MATCH_CLOSURE_RATIO,
        border_touch_ratio=MATCH_BORDER_TOUCH_RATIO,
    )
    print(f"[CLOSED] {len(closed_matches)} durchgängig umrandete Pakete extrahiert (Input für SAM3D)")

    # Phase 3c: SAM3D auf den geschlossenen Stufe-6-Masken
    sam3d_masks, sam3d_labels, sam3d_boxes = [], [], []
    if closed_matches:
        s6_masks = [m["mask"] for m in closed_matches]
        s6_boxes = [m["matched_box"] for m in closed_matches]
        s6_labels = [m["label"] for m in closed_matches]
        s6_scores = [1.0] * len(closed_matches)
        sam3d_masks, sam3d_boxes, _, sam3d_labels = refine_masks_3d(
            s6_masks, s6_boxes, s6_scores, s6_labels, session_path,
            session_context=session_context,
        )
    else:
        print("[SAM3D] Übersprungen – keine geschlossenen Pakete als Input.")

    # Phase 3.5: Bottom-plane inference on SAM3D masks
    candidates = []
    if sam3d_masks:
        bottom_cfg = load_bottom_inference_config()
        candidates = build_candidates_from_sam3d(
            sam3d_masks,
            sam3d_labels,
            session_context.depth_abs,
            session_context.plane_model,
            sam3d_boxes=sam3d_boxes,
            session_context=session_context,
        )
        stride = int(bottom_cfg.get("scene_pcd_stride", 4))
        scene_pcd = build_scene_pcd_from_depth(
            session_context.depth_abs,
            workspace_mask=session_context.workspace_mask,
            stride=stride,
        )
        pallet_plane = tuple(float(x) for x in session_context.plane_model)
        candidates = infer_bottom_planes(candidates, scene_pcd, pallet_plane, bottom_cfg)
        method_counts: dict[str, int] = {}
        for c in candidates:
            method = c.bottom.bottom_method if c.bottom else "none"
            method_counts[method] = method_counts.get(method, 0) + 1
            print(
                f"[BOTTOM] {c.candidate_id} '{c.debug.get('label')}': {method} "
                f"top={c.top_surface_height:.3f}m bottom={c.bottom.bottom_z:.3f}m "
                f"h={c.bottom.height_m:.3f}m conf={c.bottom.bottom_confidence:.2f} "
                f"({c.debug.get('case_label', '?')})"
            )
        dist = ", ".join(f"{k}={v}" for k, v in sorted(method_counts.items()))
        print(f"[BOTTOM] method distribution: {dist}")

    # Phase 4: Visualisierung
    results = None
    if DEBUG:
        results = visualize_3d(
            session_path,
            refined_masks,
            refined_labels,
            sobel_viz_data,
            original_masks,
            original_labels,
            dino_debug,
            sam3d_masks=sam3d_masks if sam3d_masks else None,
            sam3d_labels=sam3d_labels if sam3d_labels else None,
            closed_matches=closed_matches,
            session_context=session_context,
            candidates=candidates if candidates else None,
        )
    
    # Phase 5: Screenshots (optional)
    screenshot_paths = []
    
    # Phase 6: LLM Orchestrator (optional)
    llm_result = None
    
    return {
        "visualization": results,
        "candidates": candidates,
    }


def main():
    """Hauptfunktion: Orchestriert die Pipeline für alle Sessions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initialisiere Pipeline auf: {device}")
    print(f"Mode: PARAMETERFREI (Otsu-basierte Tiefentrennung)")
    
    # 1. DINO Laden
    print(f"Lade DINO Modell ({DINO_MODEL_ID})...")
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)
    
    sessions = get_all_session_paths()
    for session_path in sessions:
        process_session(session_path, dino_model, dino_processor)


if __name__ == "__main__":
    main()
