"""Experiment 5: aggregate funnel rows into thesis summary tables."""

from __future__ import annotations

from collections import Counter
from typing import Any

from evaluation.exp5_funnel import SceneFunnelRow, STATUS_MAPPING
from evaluation.scene_registry import ALL_CATEGORY_ROWS


def _safe_div(num: float | int, den: float | int) -> float | None:
    if not den:
        return None
    return float(num) / float(den)


def _band_row(rows: list[SceneFunnelRow], label: str) -> dict[str, Any]:
    n = len(rows)
    stage = Counter(r.funnel_stage for r in rows)
    released = stage["released"]
    released_valid = sum(1 for r in rows if r.funnel_stage == "released" and r.released_valid)
    released_invalid = released - released_valid

    return {
        "band": label,
        "n": n,
        "no_candidates": stage["no_candidates"],
        "no_target": stage["no_target"],
        "rejected": stage["rejected"],
        "released": released,
        "released_valid": released_valid,
        "released_invalid": released_invalid,
        "yield": _safe_div(released, n) or 0.0,
        "e2e_success": _safe_div(released_valid, n) or 0.0,
        "released_reliability": _safe_div(released_valid, released),
        "false_accepts": released_invalid,
    }


def table_funnel(
    rows: list[SceneFunnelRow],
    *,
    band_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    order = band_order or list(ALL_CATEGORY_ROWS)
    out = [_band_row([r for r in rows if r.category_band == band], band) for band in order]
    soft_rows = [r for r in rows if r.is_soft_matched]
    out.append(_band_row(soft_rows, "soft"))
    out.append(_band_row(rows, "total"))
    return out


def degenerate_plane_per_band(rows: list[SceneFunnelRow]) -> dict[str, int]:
    counts: dict[str, int] = {band: 0 for band in ALL_CATEGORY_ROWS}
    counts["soft"] = 0
    counts["total"] = 0
    for row in rows:
        if not row.degenerate_plane:
            continue
        counts[row.category_band] = counts.get(row.category_band, 0) + 1
        counts["total"] += 1
        if row.is_soft_matched:
            counts["soft"] += 1
    return counts


def build_summary(
    *,
    exp3_run_id: str,
    git_commit: str | None,
    rows: list[SceneFunnelRow],
    gate_stats: dict[str, Any],
    band_order: list[str],
    runtime_s: float,
) -> dict[str, Any]:
    return {
        "meta": {
            "exp3_run_id": exp3_run_id,
            "git_commit": git_commit,
            "n_scenes": gate_stats["n_scenes"],
            "n_inferred_handover": gate_stats["n_inferred_handover"],
            "status_mapping": dict(STATUS_MAPPING),
            "raw_statuses_observed": gate_stats["raw_statuses_observed"],
            "degenerate_plane_per_band": degenerate_plane_per_band(rows),
            "runtime_s": runtime_s,
        },
        "table_funnel": table_funnel(rows, band_order=band_order),
    }
