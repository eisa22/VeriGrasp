"""Perception-only pipeline stages for Experiment 1 (D → S → M → F)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from GroundingSAM.grounding_sam import run_grounding_dino_only
from Sam3D.sam3d import refine_masks_3d
from Segmentation.pallet_scene import SessionContext, prepare_session_context
from Segmentation.sobel_refinement import apply_sobel_refinement
from Visualization.visualizer import extract_dino_gradient_masks
from config import MATCH_BORDER_TOUCH_RATIO, MATCH_CLOSURE_RATIO


@dataclass
class PerceptionArtifacts:
    status: str = "ok"
    workspace_mask: np.ndarray | None = None
    height: int = 0
    width: int = 0
    boxes_D: list[list[int]] = field(default_factory=list)
    scores_D: list[float] = field(default_factory=list)
    labels_D: list[str] = field(default_factory=list)
    masks_S: list[np.ndarray] = field(default_factory=list)
    scores_S: list[float] = field(default_factory=list)
    labels_S: list[str] = field(default_factory=list)
    masks_M: list[np.ndarray] = field(default_factory=list)
    scores_M: list[float] = field(default_factory=list)
    labels_M: list[str] = field(default_factory=list)
    masks_F: list[np.ndarray] = field(default_factory=list)
    scores_F: list[float] = field(default_factory=list)
    labels_F: list[str] = field(default_factory=list)


def convert_boxes_to_masks(boxes, height, width):
    masks = []
    for box in boxes:
        mask = np.zeros((height, width), dtype=np.uint8)
        x0, y0, x1, y1 = [int(b) for b in box]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width, x1)
        y1 = min(height, y1)
        mask[y0:y1, x0:x1] = 1
        masks.append(mask)
    return masks


def _dino_score_for_box(dino_debug: dict, dino_box: list) -> float:
    boxes = dino_debug.get("post_iou_boxes", [])
    scores = dino_debug.get("post_iou_scores", [])
    if not boxes:
        return 0.5
    best_iou = -1.0
    best_score = 0.5
    db = [int(v) for v in dino_box]
    for i, box in enumerate(boxes):
        bb = [int(v) for v in box]
        ix0, iy0 = max(db[0], bb[0]), max(db[1], bb[1])
        ix1, iy1 = min(db[2], bb[2]), min(db[3], bb[3])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = (
            (db[2] - db[0]) * (db[3] - db[1])
            + (bb[2] - bb[0]) * (bb[3] - bb[1])
            - inter
        )
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_score = float(scores[i]) if i < len(scores) else 0.5
    return best_score


def run_perception(
    session_path: str,
    dino_model=None,
    dino_processor=None,
    *,
    session_context: SessionContext | None = None,
    visualize: bool = False,
) -> PerceptionArtifacts:
    """Run DINO → Sobel → matching → SAM3D; return artefacts for all measurement points."""
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
    masks = convert_boxes_to_masks(boxes, height, width)
    original_masks = [m.copy() for m in masks]
    original_labels = labels.copy()

    refined_masks, refined_labels, refined_scores, sobel_viz_data = apply_sobel_refinement(
        session_path, masks, labels, boxes, scores=scores, session_context=session_context
    )
    out.masks_S = refined_masks
    out.labels_S = refined_labels
    out.scores_S = [float(s) for s in refined_scores]

    closed_matches, excluded_matches = extract_dino_gradient_masks(
        session_path,
        dino_debug,
        sobel_viz_data,
        closure_ratio=MATCH_CLOSURE_RATIO,
        border_touch_ratio=MATCH_BORDER_TOUCH_RATIO,
        return_excluded=True,
    )

    for m in closed_matches:
        out.masks_M.append(m["mask"])
        out.labels_M.append(str(m.get("label", "")))
        closure = float(m.get("closure", 0.0))
        dino_sc = _dino_score_for_box(dino_debug, m.get("dino_box", []))
        out.scores_M.append(max(closure, dino_sc * 0.01))

    sam3d_masks: list = []
    sam3d_labels: list = []
    if closed_matches:
        s6_masks = [m["mask"] for m in closed_matches]
        s6_boxes = [m["matched_box"] for m in closed_matches]
        s6_labels = [m["label"] for m in closed_matches]
        s6_scores = out.scores_M.copy()
        sam3d_masks, _sam3d_boxes, sam3d_scores, sam3d_labels = refine_masks_3d(
            s6_masks, s6_boxes, s6_scores, s6_labels, session_path,
            session_context=session_context,
        )
        out.masks_F = sam3d_masks
        out.labels_F = [str(l) for l in sam3d_labels]
        out.scores_F = [float(s) for s in sam3d_scores]

    if visualize:
        from Visualization.visualizer import visualize_3d

        visualize_3d(
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
        )

    return out
