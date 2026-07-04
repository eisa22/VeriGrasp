"""Experiment 2: aggregation into the thesis summary tables.

Headline statistics are n / median / P95 only (no means, except the
``mean_confidence`` diagnostic in the bottom-cue table). Signed metrics
(e_top, e_bottom) report the median of the signed value and the P95 of the
absolute value.
"""

from __future__ import annotations

import numpy as np

from evaluation.exp2_metrics import BOTTOM_CUE_BY_METHOD, BOTTOM_CUE_BY_SOURCE

RATE_LAT_MM = 30.0
RATE_THETA_DEG = (12.0, 30.0)

VISIBILITY_STRATA = (
    (">=0.9", 0.9, float("inf")),
    ("0.5-0.9", 0.5, 0.9),
    ("<0.5", float("-inf"), 0.5),
)

_CUE_ORDER = [
    *(v for v in BOTTOM_CUE_BY_SOURCE.values()),
    *(v for v in BOTTOM_CUE_BY_METHOD.values()),
]


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def _rate(values: list[float], limit: float) -> float:
    if not values:
        return 0.0
    return float(np.mean([v <= limit for v in values]))


def _centroid_stats(rows: list[dict], label: str) -> dict:
    e_lat = [r["e_lat_mm"] for r in rows]
    e_top = [r["e_top_mm_signed"] for r in rows]
    return {
        "band": label,
        "n": len(rows),
        "med_e_lat_mm": _median(e_lat),
        "p95_e_lat_mm": _p95(e_lat),
        "med_e_top_mm": _median(e_top),
        "p95_abs_e_top_mm": _p95([abs(v) for v in e_top]),
        "rate_lat_30": _rate(e_lat, RATE_LAT_MM),
    }


def table_centroid(candidate_rows: list[dict], band_order: list[str]) -> list[dict]:
    """Per band over rigid candidates, one pooled soft row, one rigid total row."""
    rigid = [r for r in candidate_rows if r["packaging_type"] == "rigid"]
    soft = [r for r in candidate_rows if r["packaging_type"] == "soft"]
    out = [
        _centroid_stats([r for r in rigid if r["category_band"] == band], band)
        for band in band_order
    ]
    out.append(_centroid_stats(soft, "soft"))
    out.append(_centroid_stats(rigid, "total"))
    return out


def _normal_stats(rows: list[dict], label: str) -> dict:
    theta = [r["theta_deg"] for r in rows]
    return {
        "band": label,
        "n": len(rows),
        "med_theta_deg": _median(theta),
        "p95_theta_deg": _p95(theta),
        "rate_theta_12": _rate(theta, RATE_THETA_DEG[0]),
        "rate_theta_30": _rate(theta, RATE_THETA_DEG[1]),
    }


def table_normal(grasp_rows: list[dict], band_order: list[str]) -> list[dict]:
    evaluated = [r for r in grasp_rows if r["status"] == "evaluated"]
    out = [
        _normal_stats([r for r in evaluated if r["category_band"] == band], band)
        for band in band_order
    ]
    out.append(_normal_stats(evaluated, "total"))
    return out


def _bottom_stats(rows: list[dict], cue: str, priority: int | None) -> dict:
    e_bot = [r["e_bottom_mm_signed"] for r in rows if r["e_bottom_mm_signed"] is not None]
    confs = [r["bottom_confidence"] for r in rows if r["bottom_confidence"] is not None]
    return {
        "cue": cue,
        "priority": priority,
        "n": len(e_bot),
        "med_e_bottom_mm": _median(e_bot),
        "p95_abs_e_bottom_mm": _p95([abs(v) for v in e_bot]),
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
    }


def table_bottom(candidate_rows: list[dict]) -> list[dict]:
    """Per cue (thesis priority order) plus total, over all matched candidates."""
    with_bottom = [r for r in candidate_rows if r["e_bottom_mm_signed"] is not None]
    out = []
    seen_cues = {r["bottom_cue"] for r in with_bottom}
    for cue, priority in _CUE_ORDER:
        if cue not in seen_cues:
            continue
        rows = [r for r in with_bottom if r["bottom_cue"] == cue]
        out.append(_bottom_stats(rows, cue, priority))
    extra = sorted(seen_cues - {c for c, _ in _CUE_ORDER})
    for cue in extra:
        rows = [r for r in with_bottom if r["bottom_cue"] == cue]
        out.append(_bottom_stats(rows, cue, None))
    out.append(_bottom_stats(with_bottom, "total", None))
    return out


def extent_summary(candidate_rows: list[dict]) -> dict:
    def _vals(key: str) -> list[float]:
        return [r[key] for r in candidate_rows if r[key] is not None]

    yaw = _vals("yaw_err_deg")
    return {
        "med_err_long_rel": _median(_vals("ext_err_long_rel")),
        "med_err_short_rel": _median(_vals("ext_err_short_rel")),
        "med_err_height_rel": _median(_vals("ext_err_height_rel")),
        "med_yaw_err_deg": _median(yaw),
        "p95_yaw_err_deg": _p95(yaw),
    }


def visibility_strata(candidate_rows: list[dict]) -> list[dict]:
    """e_lat / e_top per visibility stratum, rigid candidates only."""
    rigid = [r for r in candidate_rows if r["packaging_type"] == "rigid"]
    out = []
    for label, lo, hi in VISIBILITY_STRATA:
        rows = [r for r in rigid if lo <= r["visibility_ratio"] < hi]
        out.append({
            "stratum": label,
            "n": len(rows),
            "med_e_lat_mm": _median([r["e_lat_mm"] for r in rows]),
            "med_e_top_mm": _median([r["e_top_mm_signed"] for r in rows]),
        })
    return out


def build_summary(
    candidate_rows: list[dict],
    grasp_rows: list[dict],
    band_order: list[str],
    meta: dict,
) -> dict:
    evaluated = [r for r in grasp_rows if r["status"] == "evaluated"]
    unmatched = [r for r in grasp_rows if r["status"] == "target_unmatched"]
    meta = dict(meta)
    meta.update({
        "n_matched_candidates": len(candidate_rows),
        "n_grasp_scenes": len(evaluated),
        "n_target_unmatched": len(unmatched),
    })
    return {
        "meta": meta,
        "table_centroid": table_centroid(candidate_rows, band_order),
        "table_normal": table_normal(grasp_rows, band_order),
        "table_bottom": table_bottom(candidate_rows),
        "extent": extent_summary(candidate_rows),
        "visibility_strata": visibility_strata(candidate_rows),
    }
