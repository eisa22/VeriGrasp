"""Grasp verification orchestrator (Stage 13).

Runs the three stages on a single grasp candidate. Cheap checks first; the
expensive deterministic plane fit and corridor scan run only after the bbox
premise holds (in cascade mode). Every check produces a continuous margin, so
the full mode yields ROC-ready records and a combined soft score.
"""

from __future__ import annotations

import numpy as np

from verification.config import load_verification_config
from verification.geometry import (
    Intrinsics,
    full_pointcloud,
    long_axis_in_plane,
    target_pointcloud,
)
from verification.stages import run_stage1, run_stage2, run_stage3
from verification.types import StageResult, VerificationResult


def _decisive(stages: list[StageResult]) -> tuple[int | None, str | None]:
    for st in stages:
        failed = st.first_failed()
        if failed is not None:
            return failed.stage, failed.name
    return None, None


def _soft_score(stages: list[StageResult], cfg: dict) -> float:
    """Weighted, normalised combination of all check margins (RQ3)."""
    sc = cfg.get("soft_score", {})
    weights = sc.get("weights", {})
    scales = sc.get("scales", {})
    num = 0.0
    den = 0.0
    for st in stages:
        for c in st.checks:
            w = float(weights.get(c.name, 1.0))
            scale = float(scales.get(c.name, 1.0)) or 1.0
            norm = float(c.margin) / scale
            # Clamp to keep one check from dominating the aggregate.
            norm = max(-3.0, min(3.0, norm))
            num += w * norm
            den += w
    return num / den if den > 0 else 0.0


def _clamp_bbox(bbox, h: int, w: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def verify_grasp(
    grasp,
    candidate,
    session_context,
    config: dict | None = None,
    p_full: np.ndarray | None = None,
) -> VerificationResult:
    """Verify a single suction grasp on its target candidate.

    Args:
        grasp: SuctionGrasp (position in camera frame, etc.).
        candidate: CandidateOut (provides bbox_2d).
        session_context: provides depth_abs, plane_model, intrinsics.
        config: verification config dict (loaded from YAML if None).
        p_full: optional precomputed full scene point cloud (camera frame).
    """
    cfg = config if config is not None else load_verification_config()
    mode = str(cfg.get("mode", "cascade"))

    intr = Intrinsics.from_session(session_context)
    depth = np.asarray(session_context.depth_abs)
    plane = tuple(float(x) for x in session_context.plane_model)
    axis = np.asarray(cfg.get("approach_axis", [0.0, 0.0, -1.0]), dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    if p_full is None:
        p_full = full_pointcloud(depth, intr)

    mask = np.asarray(candidate.mask_2d)
    p_target = target_pointcloud(depth, mask, intr)
    if p_target.size == 0:
        # Fallback: bbox region when mask is empty (e.g. legacy tests).
        p_target, _, _ = gather_bbox_points(depth, candidate.bbox_2d, intr)

    p_g = np.asarray(grasp.position, dtype=np.float64)

    # Orient the gripper so its long side follows the parcel's longer side.
    parcel_obb = None
    bottom = getattr(candidate, "bottom", None)
    if bottom is not None:
        parcel_obb = getattr(bottom, "parcel_obb", None)
    long_dir_xy = long_axis_in_plane(parcel_obb, plane)

    h, w = depth.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(candidate.bbox_2d, h, w)
    sub_depth = depth[y1:y2, x1:x2]
    from verification.geometry import gather_bbox_points

    p_bbox, n_valid, n_bbox_px = gather_bbox_points(depth, candidate.bbox_2d, intr)

    cascade = mode == "cascade"
    stages: list[StageResult] = []

    # --- Stage 1 ---
    st1 = run_stage1(p_bbox, n_valid, n_bbox_px, sub_depth, plane, cfg)
    stages.append(st1)
    z_top = st1.outputs.get("z_top")

    run_rest = st1.passed or not cascade

    # --- Stage 2 ---
    if run_rest:
        st2 = run_stage2(p_target, p_g, plane, axis, cfg, long_dir_xy=long_dir_xy)
        stages.append(st2)
        run_rest3 = st2.passed or not cascade
    else:
        st2 = None
        run_rest3 = False

    # --- Stage 3 ---
    if run_rest3:
        st3 = run_stage3(p_full, p_g, z_top, plane, cfg, long_dir_xy=long_dir_xy)
        stages.append(st3)

    all_passed = all(st.passed for st in stages)
    # In cascade mode we may have skipped stages; a skipped stage means an
    # earlier reject, so the verdict is REJECT.
    complete = len(stages) == 3
    verdict = "ACCEPT" if (all_passed and complete) else "REJECT"

    decisive_stage, decisive_check = _decisive(stages)

    soft = _soft_score(stages, cfg) if complete else None

    return VerificationResult(
        verdict=verdict,
        mode=mode,
        decisive_stage=decisive_stage,
        decisive_check=decisive_check,
        stages=stages,
        soft_score=soft,
        candidate_id=getattr(candidate, "candidate_id", None),
        grasp_rank=getattr(grasp, "rank", None),
    )
