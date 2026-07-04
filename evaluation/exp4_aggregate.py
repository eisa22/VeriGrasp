"""Experiment 4: aggregate ablation outputs for thesis tables."""

from __future__ import annotations

from typing import Any


def build_summary(
    *,
    exp3_run_id: str,
    git_commit: str | None,
    n_grasps: int,
    reconstruction_check: dict[str, Any],
    reference: dict[str, float],
    ablation: dict[str, Any],
    runtime_s: float,
) -> dict[str, Any]:
    return {
        "meta": {
            "exp3_run_id": exp3_run_id,
            "git_commit": git_commit,
            "n_grasps": n_grasps,
            "reconstruction_check": {
                "n_checked": reconstruction_check["n_checked"],
                "n_mismatch": reconstruction_check["cascade_mismatch"],
                "soft_max_diff": reconstruction_check["soft_max_diff"],
            },
            "unverifiable_semantics": "pass",
            "criterion_mapping_version": "thesis_tab_verif_mapping+v1_force_estimate",
            "runtime_s": runtime_s,
        },
        "reference": reference,
        "table_loo_criterion": ablation["table_loo_criterion"],
        "table_checks": ablation["table_checks"],
        "redundancy_matrix": ablation["redundancy_matrix"],
        "greedy_path": ablation["greedy_path"],
        "greedy_sufficient_size": ablation["greedy_sufficient_size"],
        "soft_score_loo": ablation["soft_score_loo"],
    }
