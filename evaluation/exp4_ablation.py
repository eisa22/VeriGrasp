"""Experiment 4: ablation study over persisted Experiment 3 per-check outcomes."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.exp3_metrics import BinaryRates, binary_rates, roc_auc, roc_curve
from evaluation.exp3_offline_verify import CHECK_ORDER, compute_soft_score_from_row
from evaluation.exp4_config import CRITERION_GROUPS, SENSOR_ARTIFACT_CHECKS


def check_outcome(row: pd.Series, check: str) -> bool:
    return bool(row[f"check_{check}_pass"] or row[f"check_{check}_unverifiable"])


def check_verifiable_fail(row: pd.Series, check: str) -> bool:
    return not row[f"check_{check}_unverifiable"] and not row[f"check_{check}_pass"]


def cascade_verdict_row(row: pd.Series, active_checks: Iterable[str]) -> str:
    active = list(active_checks)
    if not active:
        return "ACCEPT"
    return "ACCEPT" if all(check_outcome(row, c) for c in active) else "REJECT"


def cascade_verdict_series(df: pd.DataFrame, active_checks: Iterable[str]) -> np.ndarray:
    active = tuple(active_checks)
    return np.asarray(
        [cascade_verdict_row(row, active) == "ACCEPT" for _, row in df.iterrows()],
        dtype=bool,
    )


def reconstruct_cascade(df: pd.DataFrame) -> np.ndarray:
    return cascade_verdict_series(df, CHECK_ORDER)


def sole_reject_mask(df: pd.DataFrame, removed_checks: set[str]) -> np.ndarray:
    remaining = [c for c in CHECK_ORDER if c not in removed_checks]
    removed = [c for c in CHECK_ORDER if c in removed_checks]
    if not removed:
        return np.zeros(len(df), dtype=bool)

    def _sole(row: pd.Series) -> bool:
        if not remaining:
            return True
        if not all(check_outcome(row, c) for c in remaining):
            return False
        return any(not check_outcome(row, c) for c in removed)

    return np.asarray([_sole(row) for _, row in df.iterrows()], dtype=bool)


def run_reconstruction_gates(df: pd.DataFrame, *, soft_tol: float = 1e-9) -> dict:
    recomputed = np.where(reconstruct_cascade(df), "ACCEPT", "REJECT")
    stored = df["verdict_cascade"].astype(str).values
    cascade_mismatch = int((recomputed != stored).sum())

    soft_diff = np.abs(
        df.apply(compute_soft_score_from_row, axis=1).values - df["soft_score"].astype(float).values
    )
    soft_max_diff = float(soft_diff.max()) if len(soft_diff) else 0.0
    soft_bad = int((soft_diff > soft_tol).sum())

    return {
        "n_checked": len(df),
        "cascade_mismatch": cascade_mismatch,
        "soft_max_diff": soft_max_diff,
        "soft_bad": soft_bad,
    }


def assert_reconstruction_gates(df: pd.DataFrame) -> dict:
    stats = run_reconstruction_gates(df)
    if stats["cascade_mismatch"]:
        bad = df[
            np.where(reconstruct_cascade(df), "ACCEPT", "REJECT") != df["verdict_cascade"].astype(str).values
        ].head(5)
        raise SystemExit(
            f"Cascade reconstruction failed: {stats['cascade_mismatch']} mismatches. "
            f"Examples: {bad['scene_id'].tolist()}"
        )
    if stats["soft_bad"]:
        raise SystemExit(
            f"Soft-score reconstruction failed: max diff {stats['soft_max_diff']}, "
            f"{stats['soft_bad']} rows above 1e-9"
        )
    return stats


def rates_for_active(df: pd.DataFrame, active_checks: Iterable[str]) -> BinaryRates:
    valid = df["valid"].astype(bool).values
    accept = cascade_verdict_series(df, active_checks)
    return binary_rates(accept, valid)


def loo_metrics(
    df: pd.DataFrame,
    full: BinaryRates,
    removed_checks: set[str],
) -> dict:
    active = [c for c in CHECK_ORDER if c not in removed_checks]
    ablated = rates_for_active(df, active)
    flipped = int(sole_reject_mask(df, removed_checks).sum())
    return {
        "n_flipped": flipped,
        "delta_far": ablated.far - full.far,
        "delta_frr": ablated.frr - full.frr,
        "delta_accept_precision": ablated.accept_precision - full.accept_precision,
        "far": ablated.far,
        "frr": ablated.frr,
        "accept_rate": ablated.accept_rate,
        "accept_precision": ablated.accept_precision,
    }


def unique_catches(df: pd.DataFrame, check: str) -> int:
    invalid = ~df["valid"].astype(bool)
    count = 0
    for _, row in df[invalid].iterrows():
        if not check_outcome(row, check) and all(
            check_outcome(row, c) for c in CHECK_ORDER if c != check
        ):
            count += 1
    return count


def reject_set(df: pd.DataFrame, check: str) -> set[int]:
    idx: set[int] = set()
    for i, row in df.iterrows():
        if check_verifiable_fail(row, check):
            idx.add(int(i))
    return idx


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def verifiable_overlap_indices(df: pd.DataFrame, c1: str, c2: str) -> set[int]:
    out: set[int] = set()
    for i, row in df.iterrows():
        if not row[f"check_{c1}_unverifiable"] and not row[f"check_{c2}_unverifiable"]:
            out.add(int(i))
    return out


def redundancy_matrix(df: pd.DataFrame) -> tuple[list[str], list[list[float]]]:
    checks = list(CHECK_ORDER)
    rejects = {c: reject_set(df, c) for c in checks}
    n = len(checks)
    mat = [[0.0] * n for _ in range(n)]
    for i, c1 in enumerate(checks):
        for j, c2 in enumerate(checks):
            if i == j:
                mat[i][j] = 1.0
                continue
            overlap = verifiable_overlap_indices(df, c1, c2)
            a = {k for k in rejects[c1] if k in overlap}
            b = {k for k in rejects[c2] if k in overlap}
            mat[i][j] = jaccard(a, b)
    return checks, mat


def highest_overlap(checks: list[str], mat: list[list[float]], idx: int) -> tuple[str, float]:
    best_name = ""
    best_val = -1.0
    for j, name in enumerate(checks):
        if j == idx:
            continue
        val = mat[idx][j]
        if val > best_val or (val == best_val and (not best_name or name < best_name)):
            best_name = name
            best_val = val
    return best_name, float(best_val)


def greedy_forward_path(df: pd.DataFrame, full_far: float) -> tuple[list[dict], int]:
    valid = df["valid"].astype(bool).values
    active: list[str] = []
    remaining = set(CHECK_ORDER)
    path: list[dict] = []

    while remaining:
        best_check = None
        best_rates: BinaryRates | None = None
        for cand in sorted(remaining):
            trial = active + [cand]
            rates = binary_rates(cascade_verdict_series(df, trial), valid)
            if best_rates is None:
                best_check, best_rates = cand, rates
                continue
            if rates.far < best_rates.far - 1e-15:
                best_check, best_rates = cand, rates
            elif abs(rates.far - best_rates.far) <= 1e-15:
                if rates.frr < best_rates.frr - 1e-15:
                    best_check, best_rates = cand, rates
                elif abs(rates.frr - best_rates.frr) <= 1e-15 and cand < best_check:
                    best_check, best_rates = cand, rates
        assert best_check is not None and best_rates is not None
        active.append(best_check)
        remaining.remove(best_check)
        path.append(
            {
                "step": len(path) + 1,
                "added_check": best_check,
                "far": best_rates.far,
                "frr": best_rates.frr,
                "accept_precision": best_rates.accept_precision,
            }
        )

    sufficient = 0
    for step in path:
        if step["far"] <= full_far + 1e-15:
            sufficient = step["step"]
            break
    if sufficient == 0:
        sufficient = len(path)
    return path, sufficient


def soft_score_loo(df: pd.DataFrame) -> tuple[list[dict], float]:
    valid = df["valid"].astype(bool).values
    full_scores = df.apply(compute_soft_score_from_row, axis=1).values
    full_auc = roc_auc(roc_curve(full_scores, valid, higher_is_pass=True))
    out: list[dict] = []
    for check in CHECK_ORDER:
        scores = df.apply(lambda row, c=check: compute_soft_score_from_row(row, exclude={c}), axis=1).values
        auc = roc_auc(roc_curve(scores, valid, higher_is_pass=True))
        out.append({"check": check, "delta_auc": auc - full_auc})
    return out, full_auc


def run_ablation(df: pd.DataFrame) -> dict:
    valid = df["valid"].astype(bool).values
    full_accept = cascade_verdict_series(df, CHECK_ORDER)
    full = binary_rates(full_accept, valid)
    baseline = binary_rates(np.ones(len(df), dtype=bool), valid)

    if not np.array_equal(full_accept, reconstruct_cascade(df)):
        raise RuntimeError("internal: full layer != reconstruction before ablation")

    loo_rows: list[dict] = []
    for criterion, checks in CRITERION_GROUPS.items():
        metrics = loo_metrics(df, full, set(checks))
        loo_rows.append(
            {
                "unit_type": "criterion",
                "unit_name": criterion,
                "delta_soft_auc": "",
                **metrics,
            }
        )
    for check in CHECK_ORDER:
        metrics = loo_metrics(df, full, {check})
        loo_rows.append(
            {
                "unit_type": "check",
                "unit_name": check,
                **metrics,
                "delta_soft_auc": "",
            }
        )

    checks, mat = redundancy_matrix(df)
    table_checks: list[dict] = []
    for i, check in enumerate(checks):
        n_rejects = len(reject_set(df, check))
        partner, partner_j = highest_overlap(checks, mat, i)
        loo = loo_metrics(df, full, {check})
        table_checks.append(
            {
                "check": check,
                "n_rejects": n_rejects,
                "unique_catches": unique_catches(df, check),
                "delta_far": loo["delta_far"],
                "highest_overlap_with": partner,
                "highest_overlap_jaccard": partner_j,
                "inert": n_rejects == 0,
                "sensor_artifact_target": check in SENSOR_ARTIFACT_CHECKS,
            }
        )

    greedy_path, greedy_sufficient_size = greedy_forward_path(df, full.far)
    soft_loo, full_auc = soft_score_loo(df)
    for row in loo_rows:
        if row["unit_type"] == "check":
            match = next(x for x in soft_loo if x["check"] == row["unit_name"])
            row["delta_soft_auc"] = match["delta_auc"]

    identity_empty = rates_for_active(df, CHECK_ORDER)
    identity_all_removed = rates_for_active(df, [])

    return {
        "full": full,
        "baseline": baseline,
        "loo_rows": loo_rows,
        "table_loo_criterion": [
            {
                "criterion": r["unit_name"],
                "delta_far": r["delta_far"],
                "delta_frr": r["delta_frr"],
                "delta_precision": r["delta_accept_precision"],
                "n_flipped": r["n_flipped"],
            }
            for r in loo_rows
            if r["unit_type"] == "criterion"
        ],
        "table_checks": table_checks,
        "redundancy_matrix": {"checks": checks, "jaccard": mat},
        "greedy_path": greedy_path,
        "greedy_sufficient_size": greedy_sufficient_size,
        "soft_score_loo": soft_loo,
        "full_soft_auc": full_auc,
        "identity_empty_far": identity_empty.far,
        "identity_all_removed_far": identity_all_removed.far,
        "identity_all_removed_frr": identity_all_removed.frr,
        "identity_all_removed_accept_rate": identity_all_removed.accept_rate,
    }
