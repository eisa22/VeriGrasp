"""
main.py
Hauptpipeline für Pallet-Segmentierung.

Pipeline: DINO → Box-Masken → Sobel (parameterfrei) → Visualisierung (+ SAM3D parallel)

Batch-Test ohne Visualisierung (alle Szenen, schlankes JSON in Results/):
  python main.py --test
  python main.py --test 40      # nur die ersten 40 Szenen
  python main.py --test -40     # dasselbe (negatives N wird als Limit genutzt)

Pro Szene in Results/: dino, sam3d, bounding_box, grasp_candidate, verification (Kurzform), corridor (8 Eckpunkte).
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
from config import DEBUG, DINO_MODEL_ID, MATCH_CLOSURE_RATIO, MATCH_BORDER_TOUCH_RATIO, PROJECT_ROOT
from perception.configs.load import load_bottom_inference_config, load_suction_grasp_config
from perception.grasp_generation import compute_suction_grasps
from perception.adapter import (
    build_candidates_from_closed_matches,
    build_candidates_from_sam3d,
    build_match_neighbors,
    build_scene_pcd_from_depth,
)
from perception.bottom_inference import infer_bottom_planes
from perception.extraction_corridor import compute_extraction_corridor
from perception.selection import select_target_smallest_z
from verification import verify_grasp, load_verification_config
from verification.config import resolve_corridor_height
import argparse
import json
import torch
import numpy as np
import os
from datetime import datetime
from pathlib import Path
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


def _compute_extraction_corridor(candidate, session_context, config=None) -> dict | None:
    """Stage 12b: Entnahmekorridor aus Paket-Footprint (breiteste Stelle)."""
    if candidate is None or session_context is None:
        return None
    cfg = config if config is not None else load_verification_config()
    plane = tuple(float(x) for x in session_context.plane_model)
    margin = float(cfg.get("corridor", {}).get("safety_margin_m", 0.0))
    return compute_extraction_corridor(
        candidate,
        plane,
        lift_height_m=resolve_corridor_height(cfg),
        safety_margin_m=margin,
    )


def _run_verification(
    session_path: str,
    session_context,
    selection_result,
    grasp_result,
    extraction_corridor=None,
    config=None,
) -> tuple[dict | None, object | None]:
    """Stage 13: Deterministische Verifikation des primären Greifpunkts.

    Annotiert nur (kein Fallback): schreibt verification_result.json und gibt
    ``(payload, result)`` zurück. ``payload`` (serialisierbar) wird in
    pipeline_result.json eingebettet, ``result`` (VerificationResult) speist die
    Stufen-Visualisierung. Der gewählte Greifpunkt wird nicht ersetzt.
    """
    if (
        session_context is None
        or selection_result is None
        or selection_result.primary is None
        or grasp_result is None
    ):
        return None, None

    grasp = grasp_result.primary_grasp or (
        grasp_result.grasps[0] if grasp_result.grasps else None
    )
    if grasp is None:
        return None, None

    candidate = selection_result.primary.candidate
    cfg = config if config is not None else load_verification_config()

    try:
        result = verify_grasp(
            grasp,
            candidate,
            session_context,
            config=cfg,
            corridor=extraction_corridor,
        )
    except Exception as exc:  # verification must never crash the pipeline
        print(f"[VERIFY] WARN Verifikation fehlgeschlagen: {exc}")
        return {"verdict": "ERROR", "error": str(exc)}, None

    payload = result.to_serializable()
    out_path = os.path.join(session_path, "verification_result.json")
    try:
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        print(f"[VERIFY] Audit -> {out_path}")
    except OSError as exc:
        print(f"[VERIFY] Speichern fehlgeschlagen ({out_path}): {exc}")

    decisive = (
        f"stage{result.decisive_stage}/{result.decisive_check}"
        if result.decisive_check
        else "-"
    )
    soft = f"{result.soft_score:.3f}" if result.soft_score is not None else "n/a"
    print(
        f"[VERIFY] verdict={result.verdict} mode={result.mode} "
        f"decisive={decisive} soft_score={soft}"
    )
    for st in result.stages:
        failed = st.first_failed()
        flag = "PASS" if st.passed else f"FAIL@{failed.name}" if failed else "FAIL"
        print(f"[VERIFY]   stage{st.stage} {st.name}: {flag}")

    # Für die Visualisierung immer ALLE drei Stufen auswerten, auch wenn der
    # produktive cascade-Modus früh abbricht (sonst fehlt z. B. Stufe 3 in der
    # Anzeige). Das gespeicherte Verdikt bleibt das cascade-Ergebnis.
    viz_result = result
    if len(result.stages) < 3:
        try:
            full_cfg = {**cfg, "mode": "full"}
            viz_result = verify_grasp(
                grasp,
                candidate,
                session_context,
                config=full_cfg,
                corridor=extraction_corridor,
            )
        except Exception as exc:
            print(f"[VERIFY] WARN Full-Mode für Visualisierung fehlgeschlagen: {exc}")
            viz_result = result

    return payload, viz_result


def _save_final_handover(
    session_path: str,
    selection_result,
    grasp_result,
    verification=None,
    extraction_corridor=None,
) -> dict | None:
    """Stage 12: Konsolidierte Ausgabe — 3D/2D Bounding Box + Greifkandidat."""
    scene_name = os.path.basename(session_path.rstrip(os.sep))

    payload: dict = {
        "scene": scene_name,
        "session_path": session_path,
        "status": "incomplete",
        "bounding_box": None,
        "grasp_candidate": None,
        "extraction_corridor": extraction_corridor,
        "verification": verification,
    }

    if selection_result is None or selection_result.primary is None:
        payload["status"] = "no_target"
        out_path = os.path.join(session_path, "pipeline_result.json")
        try:
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, ensure_ascii=False)
            print(f"[OUTPUT] Kein Zielpaket — gespeichert -> {out_path}")
        except OSError as exc:
            print(f"[OUTPUT] Speichern fehlgeschlagen ({out_path}): {exc}")
        return payload

    cand = selection_result.primary.candidate
    bottom = cand.bottom
    bbox_2d = [int(v) for v in cand.bbox_2d]

    bounding_box = {
        "candidate_id": cand.candidate_id,
        "label": str(cand.debug.get("label", "")),
        "bbox_2d": bbox_2d,
        "top_surface_height_m": float(cand.top_surface_height),
        "centroid_3d": list(map(float, cand.centroid_3d.tolist())),
    }
    if bottom is not None:
        bounding_box.update({
            "bottom_z_m": float(bottom.bottom_z),
            "height_m": float(bottom.height_m),
            "bottom_method": bottom.bottom_method,
            "bottom_confidence": float(bottom.bottom_confidence),
            "parcel_obb": bottom.parcel_obb,
        })

    payload["bounding_box"] = bounding_box

    if grasp_result is None or not grasp_result.grasps:
        payload["status"] = "no_grasp"
        err = grasp_result.debug.get("error") if grasp_result else "grasp_skipped"
        payload["grasp_error"] = err
    else:
        primary = grasp_result.primary_grasp or grasp_result.grasps[0]
        payload["status"] = "success"
        payload["grasp_candidate"] = primary.to_serializable()
        payload["grasp_candidates"] = [g.to_serializable() for g in grasp_result.grasps]

    out_path = os.path.join(session_path, "pipeline_result.json")
    try:
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        print(f"[OUTPUT] Bounding box + Greifkandidat -> {out_path}")
    except OSError as exc:
        print(f"[OUTPUT] Speichern fehlgeschlagen ({out_path}): {exc}")

    print(f"[OUTPUT] --- {scene_name} ---")
    print(
        f"[OUTPUT] bbox_2d={bbox_2d}  "
        f"label='{bounding_box['label']}'  "
        f"h={bounding_box.get('height_m', float('nan')):.3f}m"
    )
    if payload["grasp_candidate"]:
        g = payload["grasp_candidate"]
        pos = g["position"]
        print(
            f"[OUTPUT] grasp score={g['score']:.3f}  "
            f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  "
            f"pixel=({g['pixel'][0]},{g['pixel'][1]})"
        )
    else:
        print(f"[OUTPUT] grasp=NONE ({payload.get('grasp_error', 'unknown')})")

    return payload


RESULTS_DIR = PROJECT_ROOT / "Results"


def _float_score(score) -> float:
    if torch.is_tensor(score):
        return float(score.item())
    return float(score)


def _serialize_dino_output(boxes, scores, labels) -> dict:
    """Finale DINO-Detektionen (nach Filter/NMS), ohne Masken."""
    return {
        "boxes": [[int(v) for v in box] for box in boxes],
        "labels": [str(label) for label in labels],
        "scores": [_float_score(s) for s in scores],
    }


def _serialize_sam3d_output(
    sam3d_boxes: list,
    sam3d_labels: list,
    sam3d_masks: list,
) -> dict:
    """SAM3D-Ausgabe: Boxen + Labels, ohne Masken."""
    return {
        "boxes": [[int(v) for v in box] for box in sam3d_boxes],
        "labels": [str(label) for label in sam3d_labels],
        "n_masks": len(sam3d_masks),
    }


def _extract_corridor(pipeline_result: dict | None) -> dict | None:
    """Entnahmekorridor aus pipeline_result (vor Verifikation berechnet)."""
    if not pipeline_result:
        return None
    corridor = pipeline_result.get("extraction_corridor")
    if not corridor:
        return None
    corners_bottom = corridor.get("corners_bottom_3d")
    corners_top = corridor.get("corners_top_3d")
    if not corners_bottom or not corners_top:
        return None
    return {
        "half_long_m": corridor.get("corridor_half_long_m"),
        "half_short_m": corridor.get("corridor_half_short_m"),
        "z_bottom_m": corridor.get("z_bottom_m", corridor.get("package_top_m")),
        "z_top_m": corridor.get("corridor_z_top_m"),
        "package_top_m": corridor.get("package_top_m"),
        "safety_corridor_height_m": corridor.get("safety_corridor_height_m"),
        "corners_bottom_3d": corners_bottom,
        "corners_top_3d": corners_top,
        "source": corridor.get("source"),
    }


def _verification_summary(verification: dict | None) -> dict | None:
    if not verification:
        return None
    keys = (
        "verdict",
        "mode",
        "decisive_stage",
        "decisive_check",
        "soft_score",
        "candidate_id",
        "grasp_rank",
        "error",
    )
    return {k: verification[k] for k in keys if k in verification}


def _build_test_session_entry(
    *,
    scene: str,
    session_path: str,
    index: int,
    status: str,
    dino: dict | None = None,
    sam3d: dict | None = None,
    pipeline_result: dict | None = None,
    error: str | None = None,
) -> dict:
    """Schlanke Test-Ausgabe pro Szene für Results/<timestamp>.json."""
    entry: dict = {
        "scene": scene,
        "session_path": session_path,
        "index": index,
        "status": status,
        "dino": dino,
        "sam3d": sam3d,
        "bounding_box": None,
        "grasp_candidate": None,
        "verification": None,
        "corridor": None,
        "error": error,
    }
    if isinstance(pipeline_result, dict):
        entry["bounding_box"] = pipeline_result.get("bounding_box")
        entry["grasp_candidate"] = pipeline_result.get("grasp_candidate")
        entry["verification"] = _verification_summary(pipeline_result.get("verification"))
        entry["corridor"] = _extract_corridor(pipeline_result)
    return entry


def _write_test_run_results(
    session_entries: list[dict],
    *,
    device: str,
    data_root: str,
    limit: int | None = None,
    n_available: int | None = None,
) -> Path:
    """Schreibt aggregierte Testergebnisse nach Results/<datum>_<uhrzeit>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now()
    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
    out_path = RESULTS_DIR / filename

    status_counts: dict[str, int] = {}
    for entry in session_entries:
        status = str(entry.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    payload = {
        "run_type": "test",
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "device": device,
        "data_root": data_root,
        "limit": limit,
        "n_available_sessions": n_available,
        "n_sessions": len(session_entries),
        "summary": status_counts,
        "sessions": session_entries,
    }

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print(f"\n[TEST] Ergebnisse gespeichert -> {out_path}")
    print(f"[TEST] Zusammenfassung: {status_counts}")
    return out_path


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


def process_session(
    session_path,
    dino_model=None,
    dino_processor=None,
    *,
    visualize: bool | None = None,
):
    """Verarbeitet eine Session durch die gesamte Pipeline."""
    show_viz = DEBUG if visualize is None else visualize
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
        match_neighbors = (
            build_match_neighbors(
                closed_matches,
                session_context.depth_abs,
                session_context.plane_model,
                "kept",
            )
            + build_match_neighbors(
                excluded_matches,
                session_context.depth_abs,
                session_context.plane_model,
                "excluded_by_dedup",
            )
        )
        candidates, scene_planes, gradient_plateaus = infer_bottom_planes(
            candidates,
            pallet_plane,
            bottom_cfg,
            depth=session_context.depth_abs,
            sobel_edges=sobel_edges,
            workspace_mask=session_context.workspace_mask,
            match_neighbors=match_neighbors,
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
            top = grasp_result.primary_grasp or grasp_result.grasps[0]
            sel = grasp_result.debug.get("primary_grasp_selection", "highest_score")
            print(
                f"[GRASP] primary ({sel}) score={top.score:.3f} "
                f"pos=({top.position[0]:.3f},{top.position[1]:.3f},{top.position[2]:.3f})"
            )
            if grasp_result.debug.get("centroid_constraint_enabled"):
                r = grasp_result.debug.get(
                    "radius_m_relaxed", grasp_result.debug.get("radius_m")
                )
                print(
                    f"[GRASP] centroid zone: radius={r:.3f}m "
                    f"anchor=({grasp_result.debug['anchor_3d'][0]:.3f},"
                    f"{grasp_result.debug['anchor_3d'][1]:.3f},"
                    f"{grasp_result.debug['anchor_3d'][2]:.3f})"
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

    extraction_corridor = None
    if selection_result and selection_result.primary and session_context is not None:
        extraction_corridor = _compute_extraction_corridor(
            selection_result.primary.candidate, session_context
        )
        if extraction_corridor is not None:
            try:
                corridor_path = os.path.join(session_path, "extraction_corridor.json")
                with open(corridor_path, "w", encoding="utf-8") as fp:
                    json.dump(extraction_corridor, fp, indent=2, ensure_ascii=False)
                print(
                    f"[CORRIDOR] Paket-Footprint "
                    f"{extraction_corridor['corridor_half_long_m']*2000:.0f}x"
                    f"{extraction_corridor['corridor_half_short_m']*2000:.0f}mm "
                    f"(halb) -> {corridor_path}"
                )
            except OSError as exc:
                print(f"[CORRIDOR] Speichern fehlgeschlagen: {exc}")

    # Stage 13: Verifikation des primären Greifpunkts (nur Annotation).
    verification, verification_result = _run_verification(
        session_path,
        session_context,
        selection_result,
        grasp_result,
        extraction_corridor=extraction_corridor,
    )

    pipeline_result = _save_final_handover(
        session_path,
        selection_result,
        grasp_result,
        verification=verification,
        extraction_corridor=extraction_corridor,
    )

    # Phase 4: Visualisierung
    results = None
    if show_viz:
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
            verification_result=verification_result,
            extraction_corridor=extraction_corridor,
        )
    
    # Phase 5: Screenshots (optional)
    screenshot_paths = []
    
    # Phase 6: LLM Orchestrator (optional)
    llm_result = None
    
    return {
        "visualization": results,
        "candidates": candidates,
        "grasp_result": grasp_result,
        "pipeline_result": pipeline_result,
        "dino": _serialize_dino_output(boxes, scores, labels),
        "sam3d": _serialize_sam3d_output(sam3d_boxes, sam3d_labels, sam3d_masks),
    }


def main():
    """Hauptfunktion: Orchestriert die Pipeline für alle Sessions."""
    parser = argparse.ArgumentParser(description="Pallet-Segmentierung Pipeline")
    parser.add_argument(
        "--test",
        nargs="?",
        const=0,
        default=None,
        type=int,
        metavar="N",
        help=(
            "Testlauf ohne Visualisierung; Ergebnis nach Results/<datum>_<uhrzeit>.json. "
            "Optional N: nur die ersten N Szenen (z. B. --test 40 oder --test -40). "
            "Ohne N: alle Szenen."
        ),
    )
    args = parser.parse_args()

    test_mode = args.test is not None
    test_limit: int | None = None
    if test_mode and args.test != 0:
        test_limit = abs(args.test)
        if test_limit <= 0:
            parser.error("--test N: N muss größer als 0 sein.")

    if test_mode:
        import config as _cfg

        _cfg.DEBUG = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initialisiere Pipeline auf: {device}")
    print(f"Mode: PARAMETERFREI (Otsu-basierte Tiefentrennung)")
    if test_mode:
        if test_limit is None:
            print("Testlauf: Visualisierung aus, alle Szenen, Ergebnis -> Results/")
        else:
            print(f"Testlauf: Visualisierung aus, erste {test_limit} Szenen, Ergebnis -> Results/")

    # 1. DINO Laden
    print(f"Lade DINO Modell ({DINO_MODEL_ID})...")
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)

    all_sessions = get_all_session_paths()
    n_available = len(all_sessions)
    sessions = all_sessions if test_limit is None else all_sessions[:test_limit]
    test_entries: list[dict] = []

    for index, session_path in enumerate(sessions, start=1):
        scene_name = os.path.basename(session_path.rstrip(os.sep))
        print(f"\n{'=' * 72}")
        print(f"[{index}/{len(sessions)}] {scene_name}")
        print(f"{'=' * 72}")

        try:
            result = process_session(
                session_path,
                dino_model,
                dino_processor,
                visualize=not test_mode,
            )
        except Exception as exc:
            print(f"[TEST] FEHLER in {scene_name}: {exc}")
            if test_mode:
                test_entries.append(
                    _build_test_session_entry(
                        scene=scene_name,
                        session_path=session_path,
                        index=index - 1,
                        status="error",
                        error=str(exc),
                    )
                )
            continue

        if test_mode:
            if result is None:
                entry = _build_test_session_entry(
                    scene=scene_name,
                    session_path=session_path,
                    index=index - 1,
                    status="skipped",
                )
            else:
                pipeline_result = result.get("pipeline_result")
                entry = _build_test_session_entry(
                    scene=scene_name,
                    session_path=session_path,
                    index=index - 1,
                    status=(
                        pipeline_result.get("status", "unknown")
                        if isinstance(pipeline_result, dict)
                        else "completed"
                    ),
                    dino=result.get("dino"),
                    sam3d=result.get("sam3d"),
                    pipeline_result=pipeline_result,
                )
            test_entries.append(entry)

    if test_mode:
        from path_utils import get_data_root

        _write_test_run_results(
            test_entries,
            device=device,
            data_root=get_data_root(),
            limit=test_limit,
            n_available=n_available,
        )


if __name__ == "__main__":
    main()
