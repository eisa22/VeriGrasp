"""Experiment 3: binary rates and ROC helpers for the verification layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryRates:
    n: int
    far: float
    frr: float
    accept_rate: float
    accept_precision: float


def _safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def binary_rates(accept: np.ndarray, valid: np.ndarray) -> BinaryRates:
    accept = np.asarray(accept, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    n = int(len(valid))
    n_valid = int(valid.sum())
    n_invalid = n - n_valid
    accepted_valid = int((accept & valid).sum())
    accepted_invalid = int((accept & ~valid).sum())
    rejected_valid = n_valid - accepted_valid
    return BinaryRates(
        n=n,
        far=_safe_div(accepted_invalid, n_invalid),
        frr=_safe_div(rejected_valid, n_valid),
        accept_rate=_safe_div(int(accept.sum()), n),
        accept_precision=_safe_div(accepted_valid, int(accept.sum())),
    )


def roc_curve(
    scores: np.ndarray,
    valid: np.ndarray,
    *,
    higher_is_pass: bool = True,
) -> dict[str, list[float]]:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    mask = np.isfinite(scores)
    scores = scores[mask]
    valid = valid[mask]
    if len(scores) == 0:
        return {"fpr": [], "tpr": [], "thresholds": []}

    order = np.argsort(scores)
    if not higher_is_pass:
        order = order[::-1]
    scores_sorted = scores[order]
    valid_sorted = valid[order]
    n_pos = int(valid.sum())
    n_neg = len(valid) - n_pos

    thresholds: list[float] = []
    fpr: list[float] = []
    tpr: list[float] = []
    unique_scores = np.unique(scores_sorted)
    if not higher_is_pass:
        unique_scores = unique_scores[::-1]

    for thr in unique_scores:
        accept = scores >= thr if higher_is_pass else scores <= thr
        br = binary_rates(accept, valid)
        thresholds.append(float(thr))
        fpr.append(br.far)
        tpr.append(1.0 - br.frr)

    if higher_is_pass:
        thresholds = [float(scores.max()) + 1.0] + thresholds + [float(scores.min()) - 1.0]
    else:
        thresholds = [float(scores.min()) - 1.0] + thresholds + [float(scores.max()) + 1.0]
    fpr = [0.0] + fpr + [1.0]
    tpr = [0.0] + tpr + [1.0]
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def roc_auc(curve: dict[str, list[float]]) -> float:
    fpr = curve.get("fpr") or []
    tpr = curve.get("tpr") or []
    if len(fpr) < 2:
        return float("nan")
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(tpr, fpr))


def layer_metrics(rows: list[dict]) -> dict:
    valid = np.asarray([bool(r["valid"]) for r in rows], dtype=bool)
    base_rate = _safe_div(int(valid.sum()), len(valid))
    n = len(valid)
    baseline = BinaryRates(n=n, far=1.0, frr=0.0, accept_rate=1.0, accept_precision=base_rate)
    accept_cascade = np.asarray([r["verdict_cascade"] == "ACCEPT" for r in rows], dtype=bool)
    cascade = binary_rates(accept_cascade, valid)
    return {
        "baseline": baseline,
        "cascade": cascade,
        "base_rate": base_rate,
    }
