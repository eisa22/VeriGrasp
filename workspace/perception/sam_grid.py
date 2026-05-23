"""SAM Grid-Prompts innerhalb DINO-ROIs (feingranulare Masken)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from perception.config_loader import SamConfig
from perception.mask_utils import ensure_mask_hw


def _box_area_ratio(box: list, width: int, height: int) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, (x1 - x0) * (y1 - y0)) / float(width * height)


def boxes_to_sam_masks_adaptive(
    rgb: np.ndarray,
    boxes: list,
    labels: list[str],
    scores: list[float],
    config: SamConfig,
    *,
    sam_model=None,
    sam_processor=None,
) -> tuple[list[np.ndarray], list[float], list[str], list]:
    """
    Kleine DINO-Boxen → ein SAM-Mask pro Box.
    Große ROIs → SAM-Grid (16×16 Sub-Boxen) wie in GroundingSAM/sam_grid_generator.py.
    """
    from GroundingSAM.sam_grid_generator import generate_sam_masks_from_roi
    from perception.sam_hf import boxes_to_sam_masks, get_sam

    if not boxes:
        return [], [], [], []

    H, W = rgb.shape[:2]
    image = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")

    if sam_model is None or sam_processor is None:
        sam_processor, sam_model, _ = get_sam(config.model)

    small_boxes, small_labels, small_scores = [], [], []
    all_masks: list[np.ndarray] = []
    all_scores: list[float] = []
    all_labels: list[str] = []
    out_boxes: list = []

    for box, label, score in zip(boxes, labels, scores):
        if _box_area_ratio(box, W, H) >= config.grid_on_roi_area_frac:
            roi_masks = generate_sam_masks_from_roi(
                sam_model,
                sam_processor,
                image,
                box,
                grid_size=config.grid_size,
                min_area=config.grid_min_area_px,
            )
            for mask in roi_masks:
                mask = ensure_mask_hw(mask, H, W)
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                out_boxes.append(
                    [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                )
                all_masks.append(mask)
                all_scores.append(float(score))
                all_labels.append(label)
        else:
            small_boxes.append(box)
            small_labels.append(label)
            small_scores.append(score)

    if small_boxes:
        small_masks = boxes_to_sam_masks(
            rgb, small_boxes, model_key=config.model, sam_model=sam_model, sam_processor=sam_processor
        )
        for mask, box, label, score in zip(small_masks, small_boxes, small_labels, small_scores):
            all_masks.append(mask)
            all_scores.append(float(score))
            all_labels.append(label)
            out_boxes.append(box)

    return all_masks, all_scores, all_labels, out_boxes


def boxes_to_sam_masks_grid(
    rgb: np.ndarray,
    boxes: list,
    labels: list[str],
    scores: list[float],
    config: SamConfig,
    *,
    sam_model=None,
    sam_processor=None,
) -> tuple[list[np.ndarray], list[float], list[str], list]:
    """Immer SAM-Grid pro DINO-ROI (maximale Feingranularität)."""
    grid_cfg = config.model_copy(update={"grid_on_roi_area_frac": 0.0})
    return boxes_to_sam_masks_adaptive(
        rgb, boxes, labels, scores, grid_cfg, sam_model=sam_model, sam_processor=sam_processor
    )
