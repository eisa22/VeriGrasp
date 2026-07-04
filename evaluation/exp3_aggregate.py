"""Experiment 3: build the thesis summary JSON."""

from __future__ import annotations

from typing import Any

from evaluation.exp3_metrics import (
    base_rate_per_band,
    check_criterion_matrix,
    check_metrics,
    criterion_violations,
    layer_metrics,
    per_band_layer,
)
from evaluation.exp3_offline_verify import CHECK_ORDER


def build_summary(
    rows: list[dict],
    band_order: list[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    table_checks = [check_metrics(rows, name) for name in CHECK_ORDER]
    table_layer, roc_soft, _ = layer_metrics(rows)
    base_rate = base_rate_per_band(rows, band_order)

    margin_orientation = {name: "higher_is_pass" for name in CHECK_ORDER}

    return {
        "meta": {
            **meta,
            "margin_orientation": margin_orientation,
        },
        "base_rate": base_rate,
        "criterion_violations": criterion_violations(rows),
        "table_checks": table_checks,
        "check_criterion_matrix": check_criterion_matrix(rows, list(CHECK_ORDER)),
        "table_layer": table_layer,
        "roc_soft_score": roc_soft,
        "per_band_layer": per_band_layer(rows, band_order),
    }
