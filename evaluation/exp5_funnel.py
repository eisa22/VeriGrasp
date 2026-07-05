"""Experiment 5: end-to-end funnel classification per scene."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.scene_registry import scene_meta

FUNNEL_STAGES = ("no_candidates", "no_target", "rejected", "released")

# Checked-in lookup: handover_status_raw -> funnel_stage
STATUS_MAPPING: dict[str, str] = {
    "no_grasp": "no_target",
    "no_target|empty_candidates": "no_candidates",
    "no_target|has_candidates": "no_target",
    "inferred|empty_candidates": "no_candidates",
    "inferred|has_candidates": "no_target",
    "inferred|no_grasp": "no_target",
    "success|ACCEPT": "released",
    "success|REJECT": "rejected",
}

EXPECTED_TOTALS = {
    "n_scenes": 728,
    "n_grasp_scenes": 616,
    "released": 389,
    "released_valid": 265,
    "released_invalid": 124,
    "rejected": 227,
    "pre_grasp": 112,
}


@dataclass(frozen=True)
class SceneFunnelRow:
    scene_id: str
    category_band: str
    funnel_stage: str
    released: bool | None
    released_valid: bool | None
    handover_status_raw: str
    degenerate_plane: bool
    gt_class: str | None = None
    is_soft_matched: bool = False


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _candidate_count(session_path: Path) -> int:
    stage8 = _load_json(session_path / "stage8_candidates.json")
    if not stage8:
        return 0
    return len(stage8.get("candidates") or [])


def _has_primary_target(session_path: Path) -> bool:
    stage10 = _load_json(session_path / "stage10_selected_target.json")
    if not stage10:
        return False
    primary = stage10.get("primary")
    if not primary:
        return False
    cand = primary.get("candidate")
    return cand is not None


def _has_primary_grasp(session_path: Path) -> bool:
    stage11 = _load_json(session_path / "stage11_suction_grasps.json")
    if not stage11:
        return False
    grasp = stage11.get("primary_grasp")
    return bool(grasp and grasp.get("normal"))


def is_degenerate_plane_fit(prep: dict) -> bool:
    """RANSAC fallback plane: |d|/||n|| < 0.5 * z_pallet (same rule as Exp 2)."""
    a, b, c, d = (float(x) for x in prep["plane_model"])
    norm = (a * a + b * b + c * c) ** 0.5 + 1e-12
    origin_dist = abs(d) / norm
    return origin_dist < 0.5 * float(prep["z_pallet_m"])


def infer_missing_handover(session_path: Path) -> str:
    """Derive handover_status_raw when pipeline_result.json is absent."""
    n_cand = _candidate_count(session_path)
    if n_cand == 0:
        return "inferred|empty_candidates"
    if not _has_primary_target(session_path):
        return "inferred|has_candidates"
    if not _has_primary_grasp(session_path):
        return "inferred|no_grasp"
    raise ValueError(
        f"{session_path.name}: inferred grasp scene without pipeline_result "
        "(expected in exp3_per_grasp.csv)"
    )


def _handover_key_from_pipeline(
    pipeline_result: dict,
    session_path: Path,
) -> str:
    status = pipeline_result.get("status")
    if status == "success":
        verdict = (pipeline_result.get("verification") or {}).get("verdict")
        if verdict not in ("ACCEPT", "REJECT"):
            raise ValueError(
                f"{session_path.name}: success without ACCEPT/REJECT verdict ({verdict!r})"
            )
        return f"success|{verdict}"
    if status == "no_grasp":
        return "no_grasp"
    if status == "no_target":
        n_cand = _candidate_count(session_path)
        suffix = "empty_candidates" if n_cand == 0 else "has_candidates"
        return f"no_target|{suffix}"
    raise ValueError(f"{session_path.name}: unmapped pipeline status {status!r}")


def funnel_stage_from_raw(handover_status_raw: str) -> str:
    stage = STATUS_MAPPING.get(handover_status_raw)
    if stage is None:
        raise ValueError(f"Unmapped handover_status_raw: {handover_status_raw!r}")
    return stage


def classify_scene(
    session_path: Path,
    *,
    exp3_row: pd.Series | None = None,
) -> SceneFunnelRow:
    meta = scene_meta(session_path)
    prep = _load_json(session_path / "stage_prep_context.json")
    degenerate = bool(prep and is_degenerate_plane_fit(prep))

    pipeline_path = session_path / "pipeline_result.json"
    if pipeline_path.exists():
        pipeline_result = _load_json(pipeline_path)
        handover_raw = _handover_key_from_pipeline(pipeline_result, session_path)
    else:
        handover_raw = infer_missing_handover(session_path)

    funnel_stage = funnel_stage_from_raw(handover_raw)

    released: bool | None = None
    released_valid: bool | None = None
    gt_class: str | None = None
    is_soft = False

    if exp3_row is not None:
        verdict = str(exp3_row["verdict_cascade"])
        funnel_stage = "released" if verdict == "ACCEPT" else "rejected"
        if pipeline_path.exists():
            pipeline_result = _load_json(pipeline_path)
            if pipeline_result and pipeline_result.get("status") == "success":
                pv = (pipeline_result.get("verification") or {}).get("verdict")
                expected = "released" if pv == "ACCEPT" else "rejected"
                if funnel_stage != expected:
                    raise ValueError(
                        f"{session_path.name}: exp3 funnel {funnel_stage!r} != "
                        f"pipeline verification {expected!r}"
                    )
        released = verdict == "ACCEPT"
        if released:
            released_valid = bool(exp3_row["valid"])
        gt_class = str(exp3_row["gt_class"]) if pd.notna(exp3_row.get("gt_class")) else None
        is_soft = bool(exp3_row.get("target_matched")) and bool(
            exp3_row.get("_is_soft_class", False)
        )
    elif funnel_stage in ("released", "rejected"):
        raise ValueError(
            f"{session_path.name}: grasp funnel stage but missing from exp3_per_grasp.csv"
        )

    return SceneFunnelRow(
        scene_id=meta.scene_id,
        category_band=meta.category,
        funnel_stage=funnel_stage,
        released=released,
        released_valid=released_valid,
        handover_status_raw=handover_raw,
        degenerate_plane=degenerate,
        gt_class=gt_class,
        is_soft_matched=is_soft,
    )


def classify_all_scenes(
    data_root: Path,
    exp3_df: pd.DataFrame,
    *,
    soft_classes: set[str],
) -> list[SceneFunnelRow]:
    exp3_by_scene = exp3_df.set_index("scene_id", drop=False)
    exp3_df = exp3_df.copy()
    exp3_df["_is_soft_class"] = exp3_df["gt_class"].isin(soft_classes)

    sessions = sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and p.name.startswith("scene_")
    )
    if len(sessions) != EXPECTED_TOTALS["n_scenes"]:
        raise ValueError(
            f"Expected {EXPECTED_TOTALS['n_scenes']} scenes under {data_root}, "
            f"found {len(sessions)}"
        )

    rows: list[SceneFunnelRow] = []
    for session_path in sessions:
        scene_id = session_path.name
        exp3_row = exp3_by_scene.loc[scene_id] if scene_id in exp3_by_scene.index else None
        if exp3_row is not None:
            exp3_row = exp3_row.copy()
            exp3_row["_is_soft_class"] = bool(
                exp3_row.get("target_matched")
                and str(exp3_row.get("gt_class", "")) in soft_classes
            )
        rows.append(classify_scene(session_path, exp3_row=exp3_row))

    return rows


def assert_funnel_gates(
    rows: list[SceneFunnelRow],
    exp3_df: pd.DataFrame,
    *,
    observed_raw_statuses: set[str] | None = None,
) -> dict[str, Any]:
    n = len(rows)
    if n != EXPECTED_TOTALS["n_scenes"]:
        raise AssertionError(f"Expected {EXPECTED_TOTALS['n_scenes']} rows, got {n}")

    stages = [r.funnel_stage for r in rows]
    if len(set(stages)) != len(stages) or len(rows) != len({r.scene_id for r in rows}):
        pass  # each scene once — checked below
    if len({r.scene_id for r in rows}) != n:
        raise AssertionError("Duplicate scene_id in funnel rows")

    stage_counts = {s: stages.count(s) for s in FUNNEL_STAGES}
    if sum(stage_counts.values()) != n:
        raise AssertionError(f"Funnel stages do not sum to {n}: {stage_counts}")

    grasp_ids = {r.scene_id for r in rows if r.funnel_stage in ("rejected", "released")}
    exp3_ids = set(exp3_df["scene_id"])
    if grasp_ids != exp3_ids:
        missing = exp3_ids - grasp_ids
        extra = grasp_ids - exp3_ids
        raise AssertionError(
            f"Grasp scene set mismatch: missing={len(missing)} extra={len(extra)}"
        )

    released_rows = [r for r in rows if r.funnel_stage == "released"]
    rejected_rows = [r for r in rows if r.funnel_stage == "rejected"]
    pre_grasp = stage_counts["no_candidates"] + stage_counts["no_target"]

    checks = {
        "n_grasp_scenes": len(grasp_ids),
        "released": len(released_rows),
        "rejected": len(rejected_rows),
        "released_valid": sum(1 for r in released_rows if r.released_valid),
        "released_invalid": sum(1 for r in released_rows if not r.released_valid),
        "pre_grasp": pre_grasp,
        "no_candidates": stage_counts["no_candidates"],
        "no_target": stage_counts["no_target"],
    }

    for key, expected in EXPECTED_TOTALS.items():
        if key == "n_scenes":
            continue
        actual = checks[key]
        if actual != expected:
            raise AssertionError(f"Expected {key}={expected}, got {actual} (full: {checks})")

    raw_seen = {r.handover_status_raw for r in rows}
    if observed_raw_statuses is not None:
        unknown = raw_seen - set(STATUS_MAPPING)
        if unknown:
            raise AssertionError(f"Unmapped raw statuses: {sorted(unknown)}")
        unlisted = raw_seen - observed_raw_statuses
        if unlisted:
            raise AssertionError(f"Raw statuses not in mapping table keys: {sorted(unlisted)}")

    for raw in raw_seen:
        if raw not in STATUS_MAPPING:
            raise AssertionError(f"Observed unmapped handover_status_raw: {raw!r}")

    n_inferred = sum(1 for r in rows if r.handover_status_raw.startswith("inferred|"))
    accept_rate = checks["released"] / checks["n_grasp_scenes"]
    accept_precision = checks["released_valid"] / checks["released"] if checks["released"] else 0.0

    return {
        **checks,
        "n_scenes": n,
        "n_inferred_handover": n_inferred,
        "accept_rate": accept_rate,
        "accept_precision": accept_precision,
        "stage_counts": stage_counts,
        "raw_statuses_observed": sorted(raw_seen),
    }
