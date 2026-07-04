"""Experiment 3: classification metrics, ROC curves, and layer-level stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.exp3_oracle import ORACLE_CRITERIA


@dataclass
class BinaryRates:
    n: int
    far: float
    frr: float
    accept_rate: float
    accept_precision: float


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def binary_rates(accept: np.ndarray, valid: np.ndarray) -> BinaryRates:
    """FAR/FRR with valid = positive class."""
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
    """Return FPR/TPR/thresholds for a score sweep (valid = positive)."""
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
        if higher_is_pass:
            accept = scores >= thr
        else:
            accept = scores <= thr
        br = binary_rates(accept, valid)
        thresholds.append(float(thr))
        fpr.append(br.far)
        tpr.append(1.0 - br.frr)

    # Endpoints for a closed ROC when useful.
    if higher_is_pass:
        thresholds = [float(scores.max()) + 1.0] + thresholds + [float(scores.min()) - 1.0]
        fpr = [0.0] + fpr + [1.0]
        tpr = [0.0] + tpr + [1.0]
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def auc_trapezoid(fpr: list[float], tpr: list[float]) -> float | None:
    if len(fpr) < 2 or len(tpr) < 2:
        return None
    f = np.asarray(fpr, dtype=np.float64)
    t = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(f)
    f, t = f[order], t[order]
    return float(np.trapezoid(t, f))


def check_metrics(
    rows: list[dict],
    check_name: str,
) -> dict[str, Any]:
    """Per-check FAR/FRR/AUC at default threshold, excluding unverifiable."""
    passed = []
    margins = []
    valid = []
    unverifiable = []
    for r in rows:
        uv = r.get(f"check_{check_name}_unverifiable")
        if uv is True or uv == "True":
            unverifiable.append(True)
            continue
        unverifiable.append(False)
        p = r.get(f"check_{check_name}_pass")
        if p == "" or p is None:
            continue
        passed.append(bool(p) if p != "False" else False)
        margins.append(float(r[f"check_{check_name}_margin"]))
        valid.append(bool(r["valid"]))

    n_uv = len(unverifiable)
    uv_share = _safe_div(sum(unverifiable), n_uv) if n_uv else 0.0
    if not passed:
        return {
            "check": check_name,
            "n": 0,
            "far": 0.0,
            "frr": 0.0,
            "auc": None,
            "unverifiable_share": uv_share,
            "degenerate": True,
        }

    accept = np.asarray(passed, dtype=bool)
    valid_arr = np.asarray(valid, dtype=bool)
    br = binary_rates(accept, valid_arr)
    margin_arr = np.asarray(margins, dtype=np.float64)
    degenerate = bool(np.allclose(margin_arr, margin_arr[0]))
    roc = roc_curve(margin_arr, valid_arr, higher_is_pass=True)
    auc = None if degenerate else auc_trapezoid(roc["fpr"], roc["tpr"])

    return {
        "check": check_name,
        "n": br.n,
        "far": br.far,
        "frr": br.frr,
        "auc": auc,
        "unverifiable_share": uv_share,
        "degenerate": degenerate,
    }


def _violated_criteria_list(row: dict) -> list[str]:
    raw = str(row.get("violated_criteria") or "").strip()
    if not raw:
        return []
    return [c.strip() for c in raw.split(";") if c.strip()]


def check_criterion_matrix(rows: list[dict], check_names: list[str]) -> list[dict]:
    out: list[dict] = []
    for check in check_names:
        for criterion in ORACLE_CRITERIA:
            violators = [r for r in rows if criterion in _violated_criteria_list(r)]
            if not violators:
                out.append({
                    "check": check,
                    "criterion": criterion,
                    "reject_share": 0.0,
                    "n": 0,
                })
                continue
            rejected = 0
            for r in violators:
                p = r.get(f"check_{check}_pass")
                uv = r.get(f"check_{check}_unverifiable")
                if uv is True or uv == "True" or p == "" or p is None:
                    continue
                if not (bool(p) if p != "False" else False):
                    rejected += 1
            n_eval = sum(
                1 for r in violators
                if r.get(f"check_{check}_pass") not in ("", None)
                and r.get(f"check_{check}_unverifiable") not in (True, "True")
            )
            out.append({
                "check": check,
                "criterion": criterion,
                "reject_share": _safe_div(rejected, n_eval),
                "n": len(violators),
            })
    return out


def layer_metrics(rows: list[dict]) -> tuple[list[dict], dict, dict]:
    """Layer-level configs plus soft-score ROC and cascade operating point."""
    valid = np.asarray([bool(r["valid"]) for r in rows], dtype=bool)
    base_rate = _safe_div(int(valid.sum()), len(valid))

    baseline = BinaryRates(
        n=len(valid),
        far=1.0,
        frr=0.0,
        accept_rate=1.0,
        accept_precision=base_rate,
    )

    cascade_accept = np.asarray(
        [str(r.get("verdict_cascade", "")).upper() == "ACCEPT" for r in rows],
        dtype=bool,
    )
    cascade = binary_rates(cascade_accept, valid)

    scores = np.asarray([float(r["soft_score"]) for r in rows], dtype=np.float64)
    roc = roc_curve(scores, valid, higher_is_pass=True)
    roc_auc = auc_trapezoid(roc["fpr"], roc["tpr"])

    # Operating point matching cascade FRR.
    target_frr = cascade.frr
    best_thr = 0.0
    best_diff = float("inf")
    best_point = {"fpr": cascade.far, "tpr": 1.0 - cascade.frr}
    unique = np.unique(scores[np.isfinite(scores)])
    for thr in unique:
        accept = scores >= thr
        br = binary_rates(accept, valid)
        diff = abs(br.frr - target_frr)
        if diff < best_diff:
            best_diff = diff
            best_thr = float(thr)
            best_point = {"fpr": br.far, "tpr": 1.0 - br.frr}

    accept_at = scores >= best_thr
    soft_at_frr = binary_rates(accept_at, valid)

    table = [
        {
            "config": "baseline",
            "accept_rate": baseline.accept_rate,
            "accept_precision": baseline.accept_precision,
            "far": baseline.far,
            "frr": baseline.frr,
            "n": baseline.n,
        },
        {
            "config": "cascade",
            "accept_rate": cascade.accept_rate,
            "accept_precision": cascade.accept_precision,
            "far": cascade.far,
            "frr": cascade.frr,
            "n": cascade.n,
        },
        {
            "config": "soft_score_at_cascade_frr",
            "threshold": best_thr,
            "accept_rate": soft_at_frr.accept_rate,
            "accept_precision": soft_at_frr.accept_precision,
            "far": soft_at_frr.far,
            "frr": soft_at_frr.frr,
            "n": soft_at_frr.n,
        },
    ]

    roc_soft = {
        "fpr": roc["fpr"],
        "tpr": roc["tpr"],
        "thresholds": roc["thresholds"],
        "auc": roc_auc if roc_auc is not None else 0.0,
        "cascade_point": {
            "fpr": cascade.far,
            "tpr": 1.0 - cascade.frr,
        },
    }
    return table, roc_soft, {"base_rate": base_rate}


def per_band_layer(rows: list[dict], band_order: list[str]) -> list[dict]:
    out: list[dict] = []
    for band in band_order:
        band_rows = [r for r in rows if r["category_band"] == band]
        if not band_rows:
            out.append({
                "band": band,
                "n": 0,
                "base_rate": 0.0,
                "cascade_accept_precision": 0.0,
                "cascade_frr": 0.0,
            })
            continue
        valid = np.asarray([bool(r["valid"]) for r in band_rows], dtype=bool)
        cascade_accept = np.asarray(
            [str(r.get("verdict_cascade", "")).upper() == "ACCEPT" for r in band_rows],
            dtype=bool,
        )
        br = binary_rates(cascade_accept, valid)
        out.append({
            "band": band,
            "n": len(band_rows),
            "base_rate": _safe_div(int(valid.sum()), len(valid)),
            "cascade_accept_precision": br.accept_precision,
            "cascade_frr": br.frr,
        })
    total_valid = np.asarray([bool(r["valid"]) for r in rows], dtype=bool)
    total_cascade = np.asarray(
        [str(r.get("verdict_cascade", "")).upper() == "ACCEPT" for r in rows],
        dtype=bool,
    )
    br_total = binary_rates(total_cascade, total_valid)
    out.append({
        "band": "total",
        "n": len(rows),
        "base_rate": _safe_div(int(total_valid.sum()), len(rows)),
        "cascade_accept_precision": br_total.accept_precision,
        "cascade_frr": br_total.frr,
    })
    return out


def criterion_violations(rows: list[dict]) -> list[dict]:
    invalid = [r for r in rows if not bool(r["valid"])]
    n_invalid = len(invalid)
    out: list[dict] = []
    for criterion in ORACLE_CRITERIA:
        n = sum(1 for r in invalid if criterion in _violated_criteria_list(r))
        out.append({
            "criterion": criterion,
            "n": n,
            "share_of_invalid": _safe_div(n, n_invalid),
        })
    return out


def base_rate_per_band(rows: list[dict], band_order: list[str]) -> dict[str, Any]:
    per_band: dict[str, float] = {}
    for band in band_order:
        band_rows = [r for r in rows if r["category_band"] == band]
        per_band[band] = _safe_div(
            sum(1 for r in band_rows if bool(r["valid"])),
            len(band_rows),
        )
    total = _safe_div(
        sum(1 for r in rows if bool(r["valid"])),
        len(rows),
    )
    return {"total": total, "per_band": per_band}
