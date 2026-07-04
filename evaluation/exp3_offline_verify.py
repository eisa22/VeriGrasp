"""Experiment 3: offline full-mode verification on persisted pipeline data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.masks import decode_masks_rle
from perception.candidate import BottomInference, CandidateOut
from perception.grasp_generation.types import SuctionGrasp
from Segmentation.pallet_scene import SessionContext, load_session_depth
from verification.config import load_verification_config
from verification.verify import verify_grasp

# Stable column order for CSV / summary (matches persisted full-mode audits).
CHECK_ORDER: tuple[str, ...] = (
    "existence",
    "top_height_match",
    "bbox_extent",
    "bbox_inlier",
    "bbox_surface_dist",
    "bbox_top_normal",
    "bbox_coverage",
    "planarity",
    "normal_angle",
    "normal_scatter",
    "suction_area",
    "edge_clearance",
    "data_gaps",
    "depth_seam",
    "surface_warp",
    "normal_alignment",
    "surface_warp_robust",
    "suction_force",
    "wrench_lever",
    "corridor_clear",
)


@dataclass
class CheckRecordFlat:
    name: str
    passed: bool
    margin: float
    unverifiable: bool


@dataclass
class SceneVerification:
    verdict_cascade: str
    decisive_check: str | None
    soft_score: float | None
    checks: dict[str, CheckRecordFlat]
    full_verdict: str
    full_soft_score: float | None


def _decode_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    arr = np.array([np.asarray(counts, dtype=np.int32)], dtype=object)
    return decode_masks_rle(arr, height, width)[0]


def session_context_from_prep(
    session_path: Path,
    prep: dict,
    depth_abs: np.ndarray,
) -> SessionContext:
    h, w = int(prep["height"]), int(prep["width"])
    ws = _decode_rle(prep["workspace_mask_rle"], h, w).astype(bool)
    depth = np.asarray(depth_abs, dtype=np.float32)
    depth_rel = depth.copy()
    depth_rel[depth <= 0] = 0.0
    depth_rel[~ws] = 0.0
    return SessionContext(
        depth_abs=depth,
        depth_rel=depth_rel,
        workspace_mask=ws,
        plane_model=np.asarray(prep["plane_model"], dtype=np.float64),
        z_pallet_m=float(prep.get("z_pallet_m", 0.0)),
        x_range=(0, w),
        fx=float(prep["fx"]),
        fy=float(prep["fy"]),
        cx=float(prep["cx"]),
        cy=float(prep["cy"]),
    )


def candidate_from_stage8(record: dict, height: int, width: int) -> CandidateOut:
    mask = _decode_rle(record["mask_rle"], height, width).astype(np.uint8)
    bottom = None
    obb = record.get("parcel_obb")
    if record.get("bottom_z_m") is not None and obb:
        bottom = BottomInference(
            bottom_z=float(record["bottom_z_m"]),
            bottom_method=str(record.get("bottom_method") or "unknown"),
            bottom_confidence=float(record.get("bottom_confidence") or 0.0),
            bottom_residual_m=0.0,
            used_neighbor_ids=[],
            height_m=float(record.get("height_m") or 0.0),
            parcel_obb=obb,
        )
    return CandidateOut(
        candidate_id=str(record["candidate_id"]),
        mask_2d=mask,
        points_3d=np.zeros((0, 3), dtype=np.float64),
        centroid_3d=np.asarray(record["centroid_3d"], dtype=np.float64),
        surface_normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        surface_area_m2=0.05,
        top_surface_height=float(record["top_surface_height_m"]),
        bbox_2d=tuple(int(v) for v in record["bbox_2d"]),
        debug={"label": record.get("label", "")},
        bottom=bottom,
    )


def grasp_from_json(grasp: dict) -> SuctionGrasp:
    return SuctionGrasp(
        score=float(grasp.get("score", 0.0)),
        normal=np.asarray(grasp["normal"], dtype=np.float64),
        position=np.asarray(grasp["position"], dtype=np.float64),
        row=int(grasp.get("pixel", [0, 0])[0]),
        col=int(grasp.get("pixel", [0, 0])[1]),
        rank=int(grasp.get("rank", 0)),
    )


def _checks_from_result(result) -> dict[str, CheckRecordFlat]:
    out: dict[str, CheckRecordFlat] = {}
    for st in result.stages:
        for c in st.checks:
            out[c.name] = CheckRecordFlat(
                name=c.name,
                passed=bool(c.passed),
                margin=float(c.margin),
                unverifiable=bool(c.detail.get("unverifiable", False)),
            )
    return out


def _checks_from_persisted(payload: dict | None) -> dict[str, CheckRecordFlat]:
    if not payload:
        return {}
    out: dict[str, CheckRecordFlat] = {}
    for st in payload.get("stages") or []:
        for c in st.get("checks") or []:
            name = str(c["name"])
            detail = c.get("detail") or {}
            out[name] = CheckRecordFlat(
                name=name,
                passed=bool(c.get("passed")),
                margin=float(c.get("margin", 0.0)),
                unverifiable=bool(detail.get("unverifiable", False)),
            )
    return out


def run_full_verification(
    session_path: Path,
    prep: dict,
    candidate_record: dict,
    grasp_json: dict,
    corridor: dict | None,
    *,
    skip_rerun: bool = False,
    persisted: dict | None = None,
) -> SceneVerification:
    """Run (or reuse) full-mode verification for one grasp."""
    cascade = persisted or {}
    verdict_cascade = str(cascade.get("verdict", "REJECT"))
    decisive_check = cascade.get("decisive_check")
    soft_score = cascade.get("soft_score")
    soft_score = float(soft_score) if soft_score is not None else None

    if skip_rerun and persisted and len(persisted.get("stages") or []) == 3:
        checks = _checks_from_persisted(persisted)
        full_soft = compute_soft_score_from_checks(checks)
        return SceneVerification(
            verdict_cascade=verdict_cascade,
            decisive_check=decisive_check,
            soft_score=soft_score,
            checks=checks,
            full_verdict=verdict_cascade,
            full_soft_score=full_soft,
        )

    depth = load_session_depth(str(session_path))
    ctx = session_context_from_prep(session_path, prep, depth)
    h, w = int(prep["height"]), int(prep["width"])
    candidate = candidate_from_stage8(candidate_record, h, w)
    grasp = grasp_from_json(grasp_json)
    cfg = {**load_verification_config(), "mode": "full"}
    result = verify_grasp(
        grasp,
        candidate,
        ctx,
        config=cfg,
        corridor=corridor,
    )
    checks = _checks_from_result(result)
    full_soft = compute_soft_score_from_checks(checks, cfg=cfg)
    return SceneVerification(
        verdict_cascade=verdict_cascade,
        decisive_check=decisive_check,
        soft_score=soft_score,
        checks=checks,
        full_verdict=result.verdict,
        full_soft_score=full_soft,
    )


def compute_soft_score_from_checks(
    checks: dict[str, CheckRecordFlat],
    cfg: dict | None = None,
) -> float:
    """Replicate pipeline ``_soft_score`` from per-check margins (full population)."""
    cfg = cfg if cfg is not None else load_verification_config()
    sc = cfg.get("soft_score", {})
    weights = sc.get("weights", {})
    scales = sc.get("scales", {})
    num = 0.0
    den = 0.0
    for name in CHECK_ORDER:
        rec = checks.get(name)
        if rec is None:
            continue
        w = float(weights.get(name, 1.0))
        scale = float(scales.get(name, 1.0)) or 1.0
        norm = float(rec.margin) / scale
        norm = max(-3.0, min(3.0, norm))
        num += w * norm
        den += w
    return num / den if den > 0 else 0.0


def compute_soft_score_from_row(row: dict, cfg: dict | None = None) -> float:
    """Soft score from a per-grasp CSV/metrics row (margin columns)."""
    checks: dict[str, CheckRecordFlat] = {}
    for name in CHECK_ORDER:
        margin = row.get(f"check_{name}_margin")
        if margin == "" or margin is None:
            continue
        passed = row.get(f"check_{name}_pass")
        uv = row.get(f"check_{name}_unverifiable")
        checks[name] = CheckRecordFlat(
            name=name,
            passed=bool(passed) if passed not in ("", None) else False,
            margin=float(margin),
            unverifiable=bool(uv) if uv not in ("", None) else False,
        )
    return compute_soft_score_from_checks(checks, cfg=cfg)


def resolve_primary_grasp_json(
    stage10: dict | None,
    stage11: dict | None,
    candidate_record: dict | None,
    *,
    allow_centroid_fallback: bool = False,
) -> dict | None:
    """Primary grasp from stage 11, or centroid fallback for target_unmatched scenes."""
    pg = (stage11 or {}).get("primary_grasp")
    if pg and pg.get("position"):
        return pg
    if not allow_centroid_fallback or not candidate_record:
        return None
    primary = (stage10 or {}).get("primary", {})
    cand = primary.get("candidate") or candidate_record
    centroid = cand.get("centroid_3d") or candidate_record.get("centroid_3d")
    if not centroid:
        return None
    return {
        "position": centroid,
        "normal": [0.0, 0.0, -1.0],
        "rank": int(pg.get("rank", 0)) if pg else 0,
        "score": float(pg.get("score", 0.0)) if pg else 0.0,
        "pixel": list(pg.get("pixel", [0, 0])) if pg else [0, 0],
    }


def verification_row_fields(checks: dict[str, CheckRecordFlat]) -> dict[str, Any]:
    """Flatten checks into CSV column keys."""
    fields: dict[str, Any] = {}
    for name in CHECK_ORDER:
        rec = checks.get(name)
        if rec is None:
            fields[f"check_{name}_pass"] = ""
            fields[f"check_{name}_margin"] = ""
            fields[f"check_{name}_unverifiable"] = ""
        else:
            fields[f"check_{name}_pass"] = rec.passed
            fields[f"check_{name}_margin"] = rec.margin
            fields[f"check_{name}_unverifiable"] = rec.unverifiable
    return fields
