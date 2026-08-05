"""Perception-Pipeline-Variante für Experiment 1: SAM3D-Pfad (D → S → M → F).

Vergleichsvariante zur Standard-Pipeline (perception/pipeline.py):
    D: Grounding DINO (identisch zur Standard-Pipeline)
    S: SAM-Masken (box-prompted, eine Maske pro DINO-Box) statt Sobel
    M: 2D-IoU-Deduplizierung der SAM-Masken statt Closure-Matching
    F: 3D-Verfeinerung (refine_masks_3d) — identisch zur Standard-Pipeline

Die Messpunkte D/S/M/F sind bewusst so gewählt, dass jede Stufe ein
direktes Gegenstück in der Standard-Pipeline hat und die per-Stage-Tabellen
von Experiment 1 eins zu eins vergleichbar bleiben.
"""

from __future__ import annotations

import numpy as np

from GroundingSAM.grounding_sam import (
    generate_sam_masks_for_boxes,
    run_grounding_dino_only,
)
from Sam3D.sam3d import deduplicate_masks_3d, refine_masks_3d
from Segmentation.pallet_scene import SessionContext, prepare_session_context
from perception.pipeline import PerceptionArtifacts


def _scores_for_kept_boxes(kept_boxes, all_boxes, all_scores) -> list[float]:
    """Ordnet den von SAM behaltenen Boxen ihre DINO-Scores zu.

    generate_sam_masks_for_boxes ersetzt Scores durch den Platzhalter 1.0;
    für ein sinnvolles AP-Ranking werden hier die DINO-Scores der
    zugehörigen Boxen wiederhergestellt (Zuordnung über Box-Koordinaten).
    """
    lookup: dict[tuple, float] = {}
    for box, score in zip(all_boxes, all_scores):
        key = tuple(float(v) for v in box)
        lookup.setdefault(key, float(score.item() if hasattr(score, "item") else score))
    return [lookup.get(tuple(float(v) for v in b), 0.5) for b in kept_boxes]


def run_perception_sam3d(
    session_path: str,
    dino_model=None,
    dino_processor=None,
    sam_model=None,
    sam_processor=None,
    *,
    session_context: SessionContext | None = None,
) -> PerceptionArtifacts:
    """DINO → SAM → Dedup → 3D-Verfeinerung; Artefakte für alle Messpunkte."""
    out = PerceptionArtifacts()

    if session_context is None:
        session_context = prepare_session_context(session_path)
    if session_context is None:
        out.status = "no_depth"
        return out

    out.workspace_mask = session_context.workspace_mask

    boxes, scores, labels, orig_image, dino_debug = run_grounding_dino_only(
        session_path, dino_model, dino_processor, session_context=session_context
    )

    post_boxes = dino_debug.get("post_iou_boxes", []) if dino_debug else []
    post_scores = dino_debug.get("post_iou_scores", []) if dino_debug else []
    post_labels = dino_debug.get("post_iou_labels", []) if dino_debug else []

    out.boxes_D = [[int(v) for v in b] for b in post_boxes]
    out.scores_D = [float(s) for s in post_scores]
    out.labels_D = [str(l) for l in post_labels]

    if not boxes:
        out.status = "no_detections"
        if orig_image is not None:
            out.width, out.height = orig_image.size
        return out

    width, height = orig_image.size
    out.width, out.height = width, height

    # ---- Stufe S: SAM-Masken (eine pro DINO-Box) --------------------------
    sam_masks, sam_boxes, _sam_scores, sam_labels = generate_sam_masks_for_boxes(
        session_path, boxes, labels, sam_model=sam_model, sam_processor=sam_processor
    )
    sam_scores = _scores_for_kept_boxes(sam_boxes, boxes, scores)

    out.masks_S = [np.asarray(m, dtype=np.uint8) for m in sam_masks]
    out.labels_S = [str(l) for l in sam_labels]
    out.scores_S = [float(s) for s in sam_scores]

    if not sam_masks:
        return out

    # ---- Stufe M: 2D-IoU-Deduplizierung ------------------------------------
    dedup_masks, dedup_boxes, dedup_scores, dedup_labels = deduplicate_masks_3d(
        out.masks_S, sam_boxes, sam_scores, sam_labels, session_path, iou_threshold=0.5
    )
    out.masks_M = dedup_masks
    out.labels_M = [str(l) for l in dedup_labels]
    out.scores_M = [float(s) for s in dedup_scores]

    if not dedup_masks:
        return out

    # ---- Stufe F: 3D-Verfeinerung (identisch zur Standard-Pipeline) --------
    f_masks, _f_boxes, f_scores, f_labels = refine_masks_3d(
        dedup_masks, dedup_boxes, dedup_scores, dedup_labels, session_path,
        session_context=session_context,
    )
    out.masks_F = f_masks
    out.labels_F = [str(l) for l in f_labels]
    out.scores_F = [float(s) for s in f_scores]

    return out
