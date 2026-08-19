"""Experiment 6: aggregate sensitivity outputs for thesis tables."""

from __future__ import annotations

from typing import Any

from evaluation.exp6_sensitivity import RangeRow
from evaluation.exp6_threshold_map import EXCLUDED_CHECKS, ThresholdSpec, mapping_table_json


def build_summary(
    *,
    exp3_run_id: str,
    git_commit: str | None,
    n_grasps: int,
    default_reproduction: dict[str, bool],
    raw_value_mapping: list[dict],
    reference: dict[str, float],
    table_ranges: list[dict],
    interaction: dict[str, Any],
    runtime_s: float,
    grid_flags: list[dict] | None = None,
    pairwise_rev: int | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "exp3_run_id": exp3_run_id,
        "git_commit": git_commit,
        "n_grasps": n_grasps,
        "default_reproduction": default_reproduction,
        "raw_value_mapping": raw_value_mapping,
        "excluded_checks": list(EXCLUDED_CHECKS),
        "runtime_s": runtime_s,
    }
    if grid_flags:
        meta["grid_flags"] = grid_flags
    if pairwise_rev is not None:
        meta["pairwise_rev"] = pairwise_rev
    return {
        "meta": meta,
        "reference": reference,
        "table_ranges": table_ranges,
        "interaction": interaction,
    }


def grid_flags_from_ranges(ranges: list[RangeRow]) -> list[dict]:
    return [
        {
            "check_param": r.check_param,
            "grid_type": r.grid_type,
            "grid_clamped": r.grid_clamped,
        }
        for r in ranges
    ]


def build_interaction_block(
    *,
    check_a: str,
    check_b: str,
    max_far_deviation: float,
    max_frr_deviation: float,
    independent: bool,
    within_range_far_deviation: float,
    within_range_frr_deviation: float,
    extra_pairs: list[dict] | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "check_a": check_a,
        "check_b": check_b,
        "max_far_deviation": max_far_deviation,
        "max_frr_deviation": max_frr_deviation,
        "independent": independent,
        "within_range_far_deviation": within_range_far_deviation,
        "within_range_frr_deviation": within_range_frr_deviation,
    }
    if extra_pairs:
        block["extra_pairwise"] = extra_pairs
    return block


def interaction_stats_to_block(
    stats: dict[str, float | bool],
    *,
    check_a: str,
    check_b: str,
) -> dict[str, Any]:
    return {
        "check_a": check_a,
        "check_b": check_b,
        "max_far_deviation": stats["max_far_deviation"],
        "max_frr_deviation": stats["max_frr_deviation"],
        "independent": stats["independent"],
        "within_range_far_deviation": stats["within_range_far_deviation"],
        "within_range_frr_deviation": stats["within_range_frr_deviation"],
    }
