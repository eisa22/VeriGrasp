"""Experiment 6: OAT threshold sweeps, operating ranges, pairwise interaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.exp3_metrics import BinaryRates, binary_rates
from evaluation.exp3_offline_verify import CHECK_ORDER
from evaluation.exp4_ablation import check_outcome, check_verifiable_fail
from evaluation.exp6_threshold_map import (
    ThresholdSpec,
    build_grid,
    build_pairwise_grid,
    build_threshold_specs,
    pass_with_unverifiable,
    spec_by_param,
)

REFERENCE_FAR = 0.4444444444444444
REFERENCE_FRR = 0.21364985163204747
REFERENCE_N_GRASPS = 616

FAR_TOL = 0.01
FRR_TOL = 0.05
INTERACTION_FAR_TOL = 0.01
INTERACTION_FRR_TOL = 0.05

FIXED_PAIRWISE = ("bbox_extent.extent_rel_dev_max", "bbox_inlier.inlier_min")


@dataclass(frozen=True)
class CurveRow:
    check_param: str
    tau: float
    tau_rel: float
    far: float
    frr: float
    accept_rate: float
    accept_precision: float
    n_rejects_by_check: int
    grid_type: str
    grid_clamped: bool


@dataclass(frozen=True)
class RangeRow:
    check_param: str
    default: float
    range_lo: float
    range_hi: float
    frr_at_half: float
    frr_at_double: float
    far_at_half: float
    far_at_double: float
    class_: str
    grid_type: str
    grid_clamped: bool


def _stored_pass_matrix(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    out = np.zeros((n, len(CHECK_ORDER)), dtype=bool)
    for j, check in enumerate(CHECK_ORDER):
        pass_col = df[f"check_{check}_pass"].astype(bool).values
        uv_col = df[f"check_{check}_unverifiable"].astype(bool).values
        out[:, j] = pass_col | uv_col
    return out


def _check_index(check_name: str) -> int:
    return CHECK_ORDER.index(check_name)


def layer_accept(pass_matrix: np.ndarray) -> np.ndarray:
    return pass_matrix.all(axis=1)


def _metrics_at_pass_matrix(df: pd.DataFrame, pass_matrix: np.ndarray) -> BinaryRates:
    valid = df["valid"].astype(bool).values
    return binary_rates(layer_accept(pass_matrix), valid)


def _sweep_one_check(
    df: pd.DataFrame,
    spec: ThresholdSpec,
    stored_pass: np.ndarray,
    valid: np.ndarray,
) -> list[CurveRow]:
    grid, clamped, grid_type = build_grid(spec)
    raw_pass = stored_pass.copy()
    j = _check_index(spec.check_name)
    rows: list[CurveRow] = []

    for tau in grid:
        pm = raw_pass.copy()
        pm[:, j] = pass_with_unverifiable(df, spec, float(tau))
        rates = binary_rates(layer_accept(pm), valid)
        verifiable = ~df[f"check_{spec.check_name}_unverifiable"].astype(bool).values
        raw = pass_with_unverifiable(df, spec, float(tau))
        n_rej = int((~raw & verifiable).sum())
        rows.append(
            CurveRow(
                check_param=spec.check_param,
                tau=float(tau),
                tau_rel=float(tau / spec.default) if spec.default else float("nan"),
                far=rates.far,
                frr=rates.frr,
                accept_rate=rates.accept_rate,
                accept_precision=rates.accept_precision,
                n_rejects_by_check=n_rej,
                grid_type=grid_type,
                grid_clamped=clamped,
            )
        )
    return rows


def _rate_at_rel(curves: list[CurveRow], rel: float) -> tuple[float, float]:
    best = min(curves, key=lambda r: abs(r.tau_rel - rel))
    return best.far, best.frr


def _classify_range(
    curves: list[CurveRow],
    spec: ThresholdSpec,
    ref_far: float,
    ref_frr: float,
    range_lo: float,
    range_hi: float,
) -> str:
    n_rej_max = max(c.n_rejects_by_check for c in curves)
    if n_rej_max == 0:
        return "unstressed"

    taus = [c.tau for c in curves]
    tau_min, tau_max = min(taus), max(taus)
    if range_lo <= tau_min + 1e-15 and range_hi >= tau_max - 1e-15:
        return "tolerant"

    half = 0.5 * spec.default
    double = 2.0 * spec.default
    if range_lo <= half + 1e-15 and range_hi >= double - 1e-15:
        return "moderate"
    return "critical"


def _operating_range(
    curves: list[CurveRow],
    spec: ThresholdSpec,
    ref_far: float,
    ref_frr: float,
) -> RangeRow:
    ok = [
        c
        for c in curves
        if c.far <= ref_far + FAR_TOL + 1e-15 and c.frr <= ref_frr + FRR_TOL + 1e-15
    ]
    if not ok:
        default_curve = min(curves, key=lambda c: abs(c.tau - spec.default))
        return RangeRow(
            check_param=spec.check_param,
            default=spec.default,
            range_lo=default_curve.tau,
            range_hi=default_curve.tau,
            frr_at_half=_rate_at_rel(curves, 0.5)[1],
            far_at_half=_rate_at_rel(curves, 0.5)[0],
            frr_at_double=_rate_at_rel(curves, 2.0)[1],
            far_at_double=_rate_at_rel(curves, 2.0)[0],
            class_="critical",
            grid_type=curves[0].grid_type,
            grid_clamped=curves[0].grid_clamped,
        )

    taus = sorted(c.tau for c in ok)
    default_idx = min(range(len(taus)), key=lambda i: abs(taus[i] - spec.default))
    lo_idx = hi_idx = default_idx
    while lo_idx > 0 and taus[lo_idx - 1] in taus:
        prev = taus[lo_idx - 1]
        if any(abs(c.tau - prev) < 1e-12 for c in ok):
            lo_idx -= 1
        else:
            break
    while hi_idx < len(taus) - 1:
        nxt = taus[hi_idx + 1]
        if any(abs(c.tau - nxt) < 1e-12 for c in ok):
            hi_idx += 1
        else:
            break

    range_lo, range_hi = taus[lo_idx], taus[hi_idx]
    cls = _classify_range(curves, spec, ref_far, ref_frr, range_lo, range_hi)
    return RangeRow(
        check_param=spec.check_param,
        default=spec.default,
        range_lo=range_lo,
        range_hi=range_hi,
        frr_at_half=_rate_at_rel(curves, 0.5)[1],
        far_at_half=_rate_at_rel(curves, 0.5)[0],
        frr_at_double=_rate_at_rel(curves, 2.0)[1],
        far_at_double=_rate_at_rel(curves, 2.0)[0],
        class_=cls,
        grid_type=curves[0].grid_type,
        grid_clamped=curves[0].grid_clamped,
    )


def _contiguous_range_from_default(curves: list[CurveRow], spec: ThresholdSpec, ref_far: float, ref_frr: float) -> tuple[float, float]:
    """Maximal contiguous grid interval containing default within FAR/FRR bounds."""
    sorted_curves = sorted(curves, key=lambda c: c.tau)
    taus = [c.tau for c in sorted_curves]
    ok_mask = [
        c.far <= ref_far + FAR_TOL + 1e-15 and c.frr <= ref_frr + FRR_TOL + 1e-15
        for c in sorted_curves
    ]
    default_i = min(range(len(taus)), key=lambda i: abs(taus[i] - spec.default))
    if not ok_mask[default_i]:
        return taus[default_i], taus[default_i]

    lo = hi = default_i
    while lo > 0 and ok_mask[lo - 1]:
        lo -= 1
    while hi < len(taus) - 1 and ok_mask[hi + 1]:
        hi += 1
    return taus[lo], taus[hi]


def operating_range_row(curves: list[CurveRow], spec: ThresholdSpec, ref_far: float, ref_frr: float) -> RangeRow:
    range_lo, range_hi = _contiguous_range_from_default(curves, spec, ref_far, ref_frr)
    cls = _classify_range(curves, spec, ref_far, ref_frr, range_lo, range_hi)
    return RangeRow(
        check_param=spec.check_param,
        default=spec.default,
        range_lo=range_lo,
        range_hi=range_hi,
        frr_at_half=_rate_at_rel(curves, 0.5)[1],
        far_at_half=_rate_at_rel(curves, 0.5)[0],
        frr_at_double=_rate_at_rel(curves, 2.0)[1],
        far_at_double=_rate_at_rel(curves, 2.0)[0],
        class_=cls,
        grid_type=curves[0].grid_type,
        grid_clamped=curves[0].grid_clamped,
    )


def run_oat_sweeps(df: pd.DataFrame, specs: list[ThresholdSpec] | None = None) -> tuple[list[CurveRow], list[RangeRow], BinaryRates]:
    specs = specs if specs is not None else build_threshold_specs()
    stored_pass = _stored_pass_matrix(df)
    valid = df["valid"].astype(bool).values
    ref = _metrics_at_pass_matrix(df, stored_pass)

    all_curves: list[CurveRow] = []
    all_ranges: list[RangeRow] = []
    for spec in specs:
        curves = _sweep_one_check(df, spec, stored_pass, valid)
        all_curves.extend(curves)
        all_ranges.append(operating_range_row(curves, spec, ref.far, ref.frr))
    return all_curves, all_ranges, ref


def run_pairwise_sweep(
    df: pd.DataFrame,
    spec_a: ThresholdSpec,
    spec_b: ThresholdSpec,
    stored_pass: np.ndarray | None = None,
) -> list[dict]:
    stored_pass = stored_pass if stored_pass is not None else _stored_pass_matrix(df)
    valid = df["valid"].astype(bool).values
    ja = _check_index(spec_a.check_name)
    jb = _check_index(spec_b.check_name)
    grid_a = build_pairwise_grid(spec_a)
    grid_b = build_pairwise_grid(spec_b)
    rows: list[dict] = []
    for ta in grid_a:
        for tb in grid_b:
            pm = stored_pass.copy()
            pm[:, ja] = pass_with_unverifiable(df, spec_a, float(ta))
            pm[:, jb] = pass_with_unverifiable(df, spec_b, float(tb))
            rates = binary_rates(layer_accept(pm), valid)
            rows.append({
                "check_a": spec_a.check_param,
                "tau_a": float(ta),
                "check_b": spec_b.check_param,
                "tau_b": float(tb),
                "far": rates.far,
                "frr": rates.frr,
            })
    return rows


def _interaction_deviation_stats(
    pairwise_rows: list[dict],
    oat_curves: list[CurveRow],
    spec_a: ThresholdSpec,
    spec_b: ThresholdSpec,
    *,
    range_a: tuple[float, float] | None = None,
    range_b: tuple[float, float] | None = None,
) -> dict[str, float | bool]:
    oat_lookup_a = [c for c in oat_curves if c.check_param == spec_a.check_param]
    oat_lookup_b = [c for c in oat_curves if c.check_param == spec_b.check_param]

    max_far_dev = 0.0
    max_frr_dev = 0.0
    within_far_dev = 0.0
    within_frr_dev = 0.0
    n_within = 0

    for row in pairwise_rows:
        if row["check_a"] != spec_a.check_param or row["check_b"] != spec_b.check_param:
            continue
        ta, tb = float(row["tau_a"]), float(row["tau_b"])
        ca = min(oat_lookup_a, key=lambda c: abs(c.tau - ta))
        cb = min(oat_lookup_b, key=lambda c: abs(c.tau - tb))
        pred_far = max(ca.far, cb.far)
        pred_frr = max(ca.frr, cb.frr)
        far_dev = abs(row["far"] - pred_far)
        frr_dev = abs(row["frr"] - pred_frr)
        max_far_dev = max(max_far_dev, far_dev)
        max_frr_dev = max(max_frr_dev, frr_dev)

        in_range = True
        if range_a is not None:
            in_range = in_range and range_a[0] - 1e-15 <= ta <= range_a[1] + 1e-15
        if range_b is not None:
            in_range = in_range and range_b[0] - 1e-15 <= tb <= range_b[1] + 1e-15
        if in_range:
            n_within += 1
            within_far_dev = max(within_far_dev, far_dev)
            within_frr_dev = max(within_frr_dev, frr_dev)

    independent = max_far_dev <= INTERACTION_FAR_TOL + 1e-15 and max_frr_dev <= INTERACTION_FRR_TOL + 1e-15
    return {
        "max_far_deviation": max_far_dev,
        "max_frr_deviation": max_frr_dev,
        "independent": independent,
        "within_range_far_deviation": within_far_dev if n_within else 0.0,
        "within_range_frr_deviation": within_frr_dev if n_within else 0.0,
    }


def interaction_deviation(
    pairwise_rows: list[dict],
    oat_curves: list[CurveRow],
    spec_a: ThresholdSpec,
    spec_b: ThresholdSpec,
    *,
    range_a: tuple[float, float] | None = None,
    range_b: tuple[float, float] | None = None,
) -> dict[str, float | bool]:
    return _interaction_deviation_stats(
        pairwise_rows,
        oat_curves,
        spec_a,
        spec_b,
        range_a=range_a,
        range_b=range_b,
    )


def range_tuple_from_table(table_ranges: list[dict], check_param: str) -> tuple[float, float]:
    row = next(r for r in table_ranges if r["check_param"] == check_param)
    return float(row["range_lo"]), float(row["range_hi"])


def narrowest_pair_specs(ranges: list[RangeRow], specs: list[ThresholdSpec]) -> tuple[ThresholdSpec, ThresholdSpec]:
    sb = spec_by_param(specs)
    scored: list[tuple[float, str]] = []
    for r in ranges:
        if r.class_ == "unstressed":
            continue
        width = r.range_hi - r.range_lo
        if r.default > 0:
            width_rel = width / r.default
        else:
            width_rel = width
        scored.append((width_rel, r.check_param))
    scored.sort(key=lambda x: (x[0], x[1]))
    if len(scored) < 2:
        scored = [(r.range_hi - r.range_lo, r.check_param) for r in ranges[:2]]
        scored.sort(key=lambda x: (x[0], x[1]))
    a_param, b_param = scored[0][1], scored[1][1]
    return sb[a_param], sb[b_param]


def assert_monotonicity(curves: list[CurveRow], spec: ThresholdSpec) -> None:
    """Tightening threshold must not decrease rejects; loosening must not increase."""
    by_tau = sorted(
        [c for c in curves if c.check_param == spec.check_param],
        key=lambda c: c.tau,
    )
    if len(by_tau) < 2:
        return

    def tightens(prev_tau: float, next_tau: float) -> bool | None:
        if spec.direction in ("pass_if_leq", "pass_if_lt"):
            return next_tau < prev_tau
        return next_tau > prev_tau

    for i in range(len(by_tau) - 1):
        a, b = by_tau[i], by_tau[i + 1]
        t = tightens(a.tau, b.tau)
        if t is True and b.n_rejects_by_check < a.n_rejects_by_check:
            raise AssertionError(
                f"Monotonicity violation (tighten) {spec.check_param}: "
                f"tau {a.tau}->{b.tau} rejects {a.n_rejects_by_check}->{b.n_rejects_by_check}"
            )
        if t is False and b.n_rejects_by_check > a.n_rejects_by_check:
            raise AssertionError(
                f"Monotonicity violation (loosen) {spec.check_param}: "
                f"tau {a.tau}->{b.tau} rejects {a.n_rejects_by_check}->{b.n_rejects_by_check}"
            )


def assert_default_reproduction(ref: BinaryRates) -> None:
    if ref.far != REFERENCE_FAR:
        raise AssertionError(f"FAR mismatch: {ref.far} vs {REFERENCE_FAR}")
    if ref.frr != REFERENCE_FRR:
        raise AssertionError(f"FRR mismatch: {ref.frr} vs {REFERENCE_FRR}")
    if ref.n != REFERENCE_N_GRASPS:
        raise AssertionError(f"n_grasps mismatch: {ref.n} vs {REFERENCE_N_GRASPS}")


def assert_grids_contain_default(curves: list[CurveRow], specs: list[ThresholdSpec]) -> None:
    sb = spec_by_param(specs)
    for spec in specs:
        taus = [c.tau for c in curves if c.check_param == spec.check_param]
        if not any(abs(t - spec.default) < 1e-9 for t in taus):
            raise AssertionError(f"Grid missing default for {spec.check_param}")


def curve_rows_to_dicts(rows: list[CurveRow]) -> list[dict]:
    return [
        {
            "check_param": r.check_param,
            "tau": r.tau,
            "tau_rel": r.tau_rel,
            "far": r.far,
            "frr": r.frr,
            "accept_rate": r.accept_rate,
            "accept_precision": r.accept_precision,
            "n_rejects_by_check": r.n_rejects_by_check,
        }
        for r in rows
    ]


def range_rows_to_dicts(rows: list[RangeRow]) -> list[dict]:
    return [
        {
            "check_param": r.check_param,
            "default": r.default,
            "range_lo": r.range_lo,
            "range_hi": r.range_hi,
            "frr_at_half": r.frr_at_half,
            "frr_at_double": r.frr_at_double,
            "far_at_half": r.far_at_half,
            "far_at_double": r.far_at_double,
            "class": r.class_,
        }
        for r in rows
    ]
