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
from perception.configs.load import load_bottom_inference_config, load_suction_grasp_config
from perception.grasp_generation import compute_suction_grasps
from perception.adapter import (
    build_candidates_from_closed_matches,
    build_candidates_from_sam3d,
    build_scene_pcd_from_depth,
)
from perception.bottom_inference import infer_bottom_planes
from perception.selection import select_target_smallest_z
import json
import torch
import numpy as np
import os
from transformers import (
    AutoProcessor as DinoProcessor,
    AutoModelForZeroShotObjectDetection as DinoModel
)


def _save_stage5_matches(session_path: str, kept: list, excluded: list) -> None:
    """Persist Stage-5 matches (all packages, kept + excluded) to JSON.

    The full mask is omitted from JSON (too large); the matched_box, label,
    z_plane and dedup status are kept. The full data stays in memory and is
    used by Stage 8 for bottom-plane inference.
    """
    def _serialize(match: dict, status: str) -> dict:
        zs = match.get("z_stats") or {}
        return {
            "label": str(match.get("label", "")),
            "status": status,
            "matched_box": [int(v) for v in match.get("matched_box", [])],
            "dino_box": [int(v) for v in match.get("dino_box", [])],
            "z_plane_m": float(zs.get("z_plane_m", float("nan"))),
            "z_plane_mm": float(match.get("z_plane_mm", float("nan"))),
            "closure": float(match.get("closure", 0.0)),
            "border_ratio": float(match.get("border_ratio", 0.0)),
            "segment_pixels": int(match.get("segment_pixels", 0)),
            "occluded_by": match.get("_occluded_by"),
        }

    payload = {
        "kept": [_serialize(m, "kept") for m in kept],
        "excluded": [_serialize(m, "excluded_by_dedup") for m in excluded],
        "n_total": len(kept) + len(excluded),
    }
    out_path = os.path.join(session_path, "stage5_matches.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[STAGE-5] gespeichert -> {out_path}")
    except OSError as exc:
        print(f"[STAGE-5] Speichern fehlgeschlagen ({out_path}): {exc}")


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
    # Auch die durch Overlap-Dedup ausgeschlossenen Matches zurückbekommen –
    # sie liefern den Top-Höhen-Hinweis für die Bottom-Inferenz (Stage 8).
    closed_matches, excluded_matches = extract_dino_gradient_masks(
        session_path,
        dino_debug,
        sobel_viz_data,
        closure_ratio=MATCH_CLOSURE_RATIO,
        border_touch_ratio=MATCH_BORDER_TOUCH_RATIO,
        return_excluded=True,
    )
    print(
        f"[STAGE-5] {len(closed_matches)} kept + {len(excluded_matches)} excluded "
        f"= {len(closed_matches) + len(excluded_matches)} Pakete insgesamt erkannt"
    )
    _save_stage5_matches(session_path, closed_matches, excluded_matches)

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

    # Phase 3.5 (Stage 8): Bottom-plane inference auf SAM3D-Masken.
    # Nachbar-Bestimmung rein gradient-basiert: Im Umfeld jeder Maske werden
    # über Sobel-Kanten Plateaus segmentiert und das höchste qualifizierende
    # Plateau (Top unter z_visible_min) liefert den Box-Boden.
    candidates = []
    scene_planes = []
    gradient_plateaus = []
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
        pallet_plane = tuple(float(x) for x in session_context.plane_model)
        sobel_edges = sobel_viz_data.get("edges") if sobel_viz_data else None
        candidates, scene_planes, gradient_plateaus = infer_bottom_planes(
            candidates,
            pallet_plane,
            bottom_cfg,
            depth=session_context.depth_abs,
            sobel_edges=sobel_edges,
            workspace_mask=session_context.workspace_mask,
            return_scene_planes=True,
            return_gradient_catalog=True,
        )
        method_counts: dict[str, int] = {}
        for c in candidates:
            method = c.bottom.bottom_method if c.bottom else "none"
            method_counts[method] = method_counts.get(method, 0) + 1
            n_pl = c.debug.get("gradient_n_plateaus", 0)
            print(
                f"[BOTTOM] {c.candidate_id} '{c.debug.get('label')}': {method} "
                f"top={c.top_surface_height:.3f}m bottom={c.bottom.bottom_z:.3f}m "
                f"h={c.bottom.height_m:.3f}m conf={c.bottom.bottom_confidence:.2f} "
                f"src={c.debug.get('neighbor_source', '-')} plateaus={n_pl} "
                f"({c.debug.get('case_label', '?')})"
            )
        dist = ", ".join(f"{k}={v}" for k, v in sorted(method_counts.items()))
        print(f"[BOTTOM] method distribution: {dist}")

    selection_result = None
    if candidates:
        selection_result = select_target_smallest_z(candidates)
        if selection_result.primary is None:
            print("[SELECT] no eligible target after bottom-inference.")
        else:
            p = selection_result.primary
            p_label = p.candidate.debug.get("label", p.candidate.candidate_id[:6])
            print(
                f"[SELECT] policy=highest_tier_smallest_extent_fewest_neighbors "
                f"max_top={selection_result.max_top_m:.3f}m "
                f"band={selection_result.top_band_m:.3f}m "
                f"neighbor_r={selection_result.neighbor_radius_m:.2f}m"
            )
            print(
                f"[SELECT] PRIMARY '{p_label}' id={p.candidate.candidate_id[:8]} "
                f"z_extent={p.score:.3f}m "
                f"peers={p.n_lateral_peers} "
                f"top={p.candidate.top_surface_height:.3f}m "
                f"bottom={p.candidate.bottom.bottom_z:.3f}m "
                f"method={p.candidate.bottom.bottom_method} "
                f"conf={p.candidate.bottom.bottom_confidence:.2f}"
            )
            print(
                f"[SELECT] top-tier ranking ({len(selection_result.ranking)} in band):"
            )
            for t in selection_result.ranking[:5]:
                lab = t.candidate.debug.get("label", t.candidate.candidate_id[:6])
                marker = "  <-- PRIMARY" if t.rank == 0 else ""
                print(
                    f"         #{t.rank}: '{lab}' top={t.candidate.top_surface_height:.3f}m "
                    f"z_extent={t.score:.3f}m peers={t.n_lateral_peers}{marker}"
                )
            if selection_result.rejected:
                below = [
                    r for _, r in selection_result.rejected if "below_top_tier" in r
                ]
                if below:
                    print(f"[SELECT] below top tier: {len(below)} parcel(s)")
            try:
                out_path = os.path.join(session_path, "stage10_selected_target.json")
                with open(out_path, "w", encoding="utf-8") as fp:
                    json.dump(selection_result.to_serializable(), fp, indent=2)
                print(f"[SELECT] wrote handover JSON: {out_path}")
            except Exception as e:
                print(f"[SELECT] WARN could not write handover JSON: {e}")

    grasp_result = None
    if selection_result and selection_result.primary and session_context is not None:
        grasp_cfg = load_suction_grasp_config()
        grasp_result = compute_suction_grasps(
            selection_result, session_context, config=grasp_cfg
        )
        n_grasps = len(grasp_result.grasps)
        print(
            f"[GRASP] backend={grasp_result.backend} "
            f"candidate={grasp_result.candidate_id[:8] if grasp_result.candidate_id else '?'} "
            f"n_grasps={n_grasps}"
        )
        if n_grasps > 0:
            top = grasp_result.grasps[0]
            print(
                f"[GRASP] best score={top.score:.3f} "
                f"pos=({top.position[0]:.3f},{top.position[1]:.3f},{top.position[2]:.3f})"
            )
            for g in grasp_result.grasps[:5]:
                print(
                    f"         #{g.rank}: score={g.score:.3f} "
                    f"pixel=({g.row},{g.col})"
                )
        elif grasp_result.debug.get("error"):
            print(f"[GRASP] WARN: {grasp_result.debug['error']}")
        try:
            grasp_path = os.path.join(session_path, "stage11_suction_grasps.json")
            with open(grasp_path, "w", encoding="utf-8") as fp:
                json.dump(grasp_result.to_serializable(), fp, indent=2)
            print(f"[GRASP] wrote handover JSON: {grasp_path}")
        except Exception as e:
            print(f"[GRASP] WARN could not write handover JSON: {e}")

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
            excluded_matches=excluded_matches,
            session_context=session_context,
            candidates=candidates if candidates else None,
            scene_planes=scene_planes if scene_planes else None,
            gradient_plateaus=gradient_plateaus if gradient_plateaus else None,
            selection_result=selection_result,
            grasp_result=grasp_result,
        )
    
    # Phase 5: Screenshots (optional)
    screenshot_paths = []
    
    # Phase 6: LLM Orchestrator (optional)
    llm_result = None
    
    return {
        "visualization": results,
        "candidates": candidates,
        "grasp_result": grasp_result,
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
