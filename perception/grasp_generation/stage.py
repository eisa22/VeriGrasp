"""Stage 11: suction grasp points for the selected target."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from perception.grasp_generation.camera import camera_from_shape
from perception.grasp_generation.centroid import (
    compute_grasp_centroid_zone,
    pick_grasp_nearest_centroid,
)
from perception.grasp_generation.types import SuctionGrasp
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

    anchor_3d, radius_m, centroid_debug = compute_grasp_centroid_zone(
        candidate, mask, depth, camera, cfg
    )

    if backend == "normal_std":
        grasps, debug = run_normal_std(
            depth,
            mask,
            camera,
            cfg,
            centroid_anchor=anchor_3d,
            centroid_radius_m=radius_m,
            centroid_debug=centroid_debug,
        )
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

    primary_grasp: SuctionGrasp | None = None
    cc = cfg.get("centroid_constraint") or {}
    pick_nearest = cc.get("pick_nearest_for_primary", True)
    if grasps:
        if (
            cc.get("enabled", False)
            and pick_nearest
            and anchor_3d is not None
        ):
            primary_grasp, near_idx = pick_grasp_nearest_centroid(
                grasps,
                anchor_3d,
                use_xy_distance=bool(cc.get("use_xy_distance", True)),
            )
            if near_idx is not None:
                pos = primary_grasp.position
                anchor = np.asarray(anchor_3d, dtype=np.float64)
                debug["primary_grasp_selection"] = "nearest_centroid"
                debug["primary_grasp_index"] = int(near_idx)
                debug["primary_grasp_dist_xy_m"] = float(
                    np.linalg.norm(pos[:2] - anchor[:2])
                )
        else:
            primary_grasp = grasps[0]
            debug["primary_grasp_selection"] = "highest_score"

    return SuctionGraspResult(
        grasps=grasps,
        candidate_id=candidate.candidate_id,
        backend=backend,
        config_snapshot=dict(cfg),
        debug=debug,
        primary_grasp=primary_grasp,
    )
