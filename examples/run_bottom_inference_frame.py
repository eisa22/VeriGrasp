#!/usr/bin/env python3
"""Run one frame through perception + Stage 2.5 and print bottom_method distribution."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForZeroShotObjectDetection as DinoModel,
    AutoProcessor as DinoProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DINO_MODEL_ID, MATCH_BORDER_TOUCH_RATIO, MATCH_CLOSURE_RATIO
from GroundingSAM.grounding_sam import run_grounding_dino_only
from Segmentation.pallet_scene import prepare_session_context
from Segmentation.sobel_refinement import apply_sobel_refinement
from Visualization.visualizer import extract_dino_gradient_masks
from main import convert_boxes_to_masks
from path_utils import get_all_session_paths
from perception.adapter import (
    build_candidates_from_closed_matches,
    build_scene_pcd_from_depth,
)
from perception.bottom_inference import infer_bottom_planes
from perception.configs.load import load_bottom_inference_config


def run_frame(session_path: str, dino_model, dino_processor) -> Counter:
    ctx = prepare_session_context(session_path)
    if ctx is None:
        raise RuntimeError(f"No depth for session: {session_path}")

    boxes, scores, labels, orig_image, dino_debug = run_grounding_dino_only(
        session_path, dino_model, dino_processor, session_context=ctx
    )
    if not boxes:
        return Counter()

    w, h = orig_image.size
    masks = convert_boxes_to_masks(boxes, h, w)
    refined_masks, refined_labels, sobel_viz = apply_sobel_refinement(
        session_path, masks, labels, boxes, session_context=ctx
    )
    closed_matches = extract_dino_gradient_masks(
        session_path,
        dino_debug,
        sobel_viz,
        closure_ratio=MATCH_CLOSURE_RATIO,
        border_touch_ratio=MATCH_BORDER_TOUCH_RATIO,
    )
    if not closed_matches:
        print("[BOTTOM] No closed matches.")
        return Counter()

    cfg = load_bottom_inference_config()
    candidates = build_candidates_from_closed_matches(
        closed_matches, ctx.depth_abs, ctx.plane_model, ctx
    )
    scene_pcd = build_scene_pcd_from_depth(
        ctx.depth_abs, workspace_mask=ctx.workspace_mask, stride=int(cfg.get("scene_pcd_stride", 4))
    )
    plane = tuple(float(x) for x in ctx.plane_model)
    candidates = infer_bottom_planes(candidates, scene_pcd, plane, cfg)

    counts = Counter(c.bottom.bottom_method for c in candidates if c.bottom)
    print(f"\nSession: {session_path}")
    print(f"Candidates: {len(candidates)}")
    for c in candidates:
        b = c.bottom
        print(
            f"  {c.candidate_id} label={c.debug.get('label')} "
            f"method={b.bottom_method} conf={b.bottom_confidence:.2f} "
            f"height={b.height_m:.3f}m case={c.debug.get('case_label')}"
        )
    print("Distribution:", dict(counts))
    return counts


def main():
    parser = argparse.ArgumentParser(description="Bottom inference on one frame")
    parser.add_argument("--session-index", type=int, default=0)
    args = parser.parse_args()

    sessions = get_all_session_paths()
    if not sessions:
        print("No sessions found.")
        return 1
    session_path = sessions[min(args.session_index, len(sessions) - 1)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)
    run_frame(session_path, model, processor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
