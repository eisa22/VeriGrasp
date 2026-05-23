"""
Originale Hauptpipeline (main.py):
  Grounding DINO → Box-Masken (Rechtecke) → Sobel/Tiefen-Refinement
Kein SAM — das war der Unterschied zur schlechteren gdino_sam-Variante.
"""

from __future__ import annotations

import time
from typing import Any

from perception.config_loader import PipelineConfig
from perception.contracts import FrameInput
from perception.legacy import apply_sobel_refinement, convert_boxes_to_masks, run_grounding_dino_only


def run_gdino_sobel(
    frame: FrameInput,
    config: PipelineConfig,
    *,
    dino_model=None,
    dino_processor=None,
    **_kwargs,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    filter_log: list[dict] = []

    t0 = time.perf_counter()
    dino_out = run_grounding_dino_only(
        frame.rgb,
        config.prompts,
        box_threshold=config.gdino.box_threshold,
        text_threshold=config.gdino.text_threshold,
        max_box_area_ratio=config.gdino.max_box_area_ratio,
        relative_iou_nms_thresh=config.gdino.relative_iou_nms_thresh,
        dino_model=dino_model,
        dino_processor=dino_processor,
    )
    timings["gdino_ms"] = (time.perf_counter() - t0) * 1000

    boxes = dino_out["boxes"]
    scores = dino_out["scores"]
    labels = dino_out["labels"]
    debug = dino_out["debug"]

    if not boxes:
        return {
            "masks": [],
            "scores": scores,
            "labels": labels,
            "boxes": [],
            "timings": timings,
            "filter_log": filter_log,
            "debug": debug,
        }

    h, w = frame.rgb.shape[:2]
    t1 = time.perf_counter()
    masks = convert_boxes_to_masks(boxes, h, w)
    timings["box_masks_ms"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    masks, labels = apply_sobel_refinement(
        frame.rgb,
        frame.depth,
        masks,
        labels=labels,
        boxes=boxes,
    )
    timings["sobel_ms"] = (time.perf_counter() - t2) * 1000

    score_list = [float(s) for s in scores[: len(masks)]]
    while len(score_list) < len(masks):
        score_list.append(1.0)

    return {
        "masks": masks,
        "scores": score_list,
        "labels": labels,
        "boxes": boxes,
        "timings": timings,
        "filter_log": filter_log,
        "debug": debug,
    }
