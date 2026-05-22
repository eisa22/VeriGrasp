"""Stage 11: suction grasp points for the selected target."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from perception.grasp_generation.camera import camera_from_shape
from perception.grasp_generation.normal_std_backend import run_normal_std
from perception.grasp_generation.types import SuctionGraspResult
from perception.selection.select_target import SelectionResult
from Segmentation.pallet_scene import SessionContext


def _align_mask_to_depth(mask: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    h, w = depth_shape
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def compute_suction_grasps(
    selection_result: SelectionResult,
    session_context: SessionContext,
    config: dict[str, Any] | None = None,
) -> SuctionGraspResult:
    """Compute vacuum suction grasp candidates for the Stage-10 primary target."""
    from perception.configs.load import load_suction_grasp_config

    cfg = config if config is not None else load_suction_grasp_config()
    backend = str(cfg.get("backend", "normal_std"))

    if selection_result.primary is None:
        return SuctionGraspResult(
            grasps=[],
            candidate_id=None,
            backend=backend,
            config_snapshot=dict(cfg),
            debug={"error": "no_primary_selection"},
        )

    candidate = selection_result.primary.candidate
    depth = session_context.depth_abs.astype(np.float32)
    h, w = depth.shape[:2]
    mask = _align_mask_to_depth(candidate.mask_2d, (h, w))
    mask = mask & (depth > 0)

    if not np.any(mask):
        return SuctionGraspResult(
            grasps=[],
            candidate_id=candidate.candidate_id,
            backend=backend,
            config_snapshot=dict(cfg),
            debug={"error": "empty_mask_after_depth_filter"},
        )

    fx = float(cfg.get("fx", 437.04))
    fy = float(cfg.get("fy", 437.04))
    camera = camera_from_shape(h, w, fx=fx, fy=fy)

    if backend == "normal_std":
        grasps, debug = run_normal_std(depth, mask, camera, cfg)
    elif backend == "neural":
        checkpoint = cfg.get("checkpoint_path")
        if not checkpoint:
            return SuctionGraspResult(
                grasps=[],
                candidate_id=candidate.candidate_id,
                backend=backend,
                config_snapshot=dict(cfg),
                debug={"error": "neural_backend_requires_checkpoint_path"},
            )
        grasps, debug = [], {"error": "neural_backend_not_implemented_use_normal_std"}
        print("[GRASP] WARN: neural backend not implemented; set backend: normal_std")
    else:
        return SuctionGraspResult(
            grasps=[],
            candidate_id=candidate.candidate_id,
            backend=backend,
            config_snapshot=dict(cfg),
            debug={"error": f"unknown_backend:{backend}"},
        )

    return SuctionGraspResult(
        grasps=grasps,
        candidate_id=candidate.candidate_id,
        backend=backend,
        config_snapshot=dict(cfg),
        debug=debug,
    )
