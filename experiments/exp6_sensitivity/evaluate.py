"""Experiment 6 — offline threshold sensitivity over Experiment 3 results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.env_snapshot import _git_commit  # noqa: E402
from evaluation.exp4_ablation import assert_reconstruction_gates  # noqa: E402
from evaluation.exp6_aggregate import (  # noqa: E402
    build_interaction_block,
    build_summary,
    grid_flags_from_ranges,
    interaction_stats_to_block,
)
from evaluation.exp6_sensitivity import (  # noqa: E402
    FIXED_PAIRWISE,
    REFERENCE_FAR,
    REFERENCE_FRR,
    assert_default_reproduction,
    assert_grids_contain_default,
    assert_monotonicity,
    curve_rows_to_dicts,
    interaction_deviation,
    narrowest_pair_specs,
    range_rows_to_dicts,
    range_tuple_from_table,
    run_oat_sweeps,
    run_pairwise_sweep,
)
from evaluation.exp6_threshold_map import (  # noqa: E402
    build_threshold_specs,
    mapping_table_json,
    spec_by_param,
    validate_round_trip,
)

CURVE_COLUMNS = [
    "check_param",
    "tau",
    "tau_rel",
    "far",
    "frr",
    "accept_rate",
    "accept_precision",
    "n_rejects_by_check",
]

PAIRWISE_COLUMNS = ["check_a", "tau_a", "check_b", "tau_b", "far", "frr"]


def _load_eval_config() -> dict:
    import yaml

    path = Path(__file__).parent / "eval_config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _write_sensitivity_figures(
    curves: list,
    ranges: list,
    out_dir: Path,
    *,
    panels_per_fig: int = 12,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from evaluation.exp6_sensitivity import CurveRow, RangeRow

    assert isinstance(curves[0], CurveRow)
    by_param: dict[str, list[CurveRow]] = {}
    for c in curves:
        by_param.setdefault(c.check_param, []).append(c)

    range_by_param = {r.check_param: r for r in ranges}
    params = sorted(by_param.keys())
    paths: list[Path] = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    chunk_starts = list(range(0, len(params), panels_per_fig))
    for fig_idx, start in enumerate(chunk_starts):
        chunk = params[start : start + panels_per_fig]
        n = len(chunk)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
        fig.suptitle("Experiment 6 — threshold sensitivity", fontsize=10, y=1.02)

        for ax_i, param in enumerate(chunk):
            ax = axes.flat[ax_i]
            pts = sorted(by_param[param], key=lambda c: c.tau_rel)
            rels = [p.tau_rel for p in pts]
            far = [p.far for p in pts]
            frr = [p.frr for p in pts]
            ax.plot(rels, far, marker="o", markersize=3, linewidth=1.0, label="FAR")
            ax.plot(rels, frr, marker="s", markersize=3, linewidth=1.0, label="FRR")
            ax.axvline(1.0, color="0.4", linestyle="--", linewidth=0.8)
            rr: RangeRow | None = range_by_param.get(param)
            if rr and rr.class_ != "unstressed":
                lo_rel = rr.range_lo / rr.default if rr.default else rr.range_lo
                hi_rel = rr.range_hi / rr.default if rr.default else rr.range_hi
                ax.axvspan(lo_rel, hi_rel, alpha=0.15, color="tab:green")
            use_log = pts[0].grid_type == "log"
            if use_log and all(r > 0 for r in rels):
                ax.set_xscale("log")
            ax.set_title(param.split(".")[-1][:18], fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            if ax_i == 0:
                ax.legend(fontsize=6, loc="best")

        for j in range(len(chunk), nrows * ncols):
            axes.flat[j].set_visible(False)

        fig.tight_layout()
        suffix = chr(ord("a") + fig_idx)
        fig_path = fig_dir / f"exp6_sensitivity_{suffix}.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(fig_path)
        print(f"[FIGURE] geschrieben -> {fig_path}")

    return paths


def main(argv: list[str] | None = None) -> None:
    cfg = _load_eval_config()
    parser = argparse.ArgumentParser(description="Experiment 6 threshold sensitivity")
    parser.add_argument(
        "--exp3-dir",
        type=Path,
        default=PROJECT_ROOT / cfg.get("exp3_dir", "Results/exp3/full_2026-07-05"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / cfg.get("out_dir", "Results/exp6/full_2026-07-05"),
    )
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    exp3_dir = args.exp3_dir.resolve()
    out_dir = args.out_dir.resolve()
    csv_path = exp3_dir / "exp3_per_grasp.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing input: {csv_path}")
    if "n_blocking_points" not in pd.read_csv(csv_path, nrows=0).columns:
        raise SystemExit(
            f"Missing n_blocking_points column — run scripts/patch_exp3_n_blocking.py first"
        )

    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    specs = build_threshold_specs()
    sb = spec_by_param(specs)

    print(f"[GATE] Exp4 cascade reconstruction on {len(df)} grasps …")
    reconstruction = assert_reconstruction_gates(df)
    print(f"[GATE] OK cascade mismatches={reconstruction['cascade_mismatch']}")

    print("[GATE] Raw-value round trip at default thresholds …")
    mismatches = validate_round_trip(df, specs)
    bad = {k: v for k, v in mismatches.items() if v > 0}
    if bad:
        raise SystemExit(f"Round-trip failed: {bad}")

    curves, ranges, ref = run_oat_sweeps(df, specs)
    assert_default_reproduction(ref)
    assert_grids_contain_default(curves, specs)
    for spec in specs:
        assert_monotonicity(curves, spec)

    table_ranges_dicts = range_rows_to_dicts(ranges)

    spec_na, spec_nb = narrowest_pair_specs(ranges, specs)
    pairwise_auto = run_pairwise_sweep(df, spec_na, spec_nb)
    stats_auto = interaction_deviation(
        pairwise_auto,
        curves,
        spec_na,
        spec_nb,
        range_a=range_tuple_from_table(table_ranges_dicts, spec_na.check_param),
        range_b=range_tuple_from_table(table_ranges_dicts, spec_nb.check_param),
    )

    spec_fe, spec_bi = sb[FIXED_PAIRWISE[0]], sb[FIXED_PAIRWISE[1]]
    pairwise_fixed = run_pairwise_sweep(df, spec_fe, spec_bi)
    stats_fixed = interaction_deviation(
        pairwise_fixed,
        curves,
        spec_fe,
        spec_bi,
        range_a=range_tuple_from_table(table_ranges_dicts, spec_fe.check_param),
        range_b=range_tuple_from_table(table_ranges_dicts, spec_bi.check_param),
    )

    all_pairwise = pairwise_auto + pairwise_fixed

    runtime_s = time.perf_counter() - t0
    summary = build_summary(
        exp3_run_id=exp3_dir.name,
        git_commit=_git_commit(),
        n_grasps=len(df),
        default_reproduction={"far_match": True, "frr_match": True},
        raw_value_mapping=mapping_table_json(specs),
        reference={"far": REFERENCE_FAR, "frr": REFERENCE_FRR},
        table_ranges=table_ranges_dicts,
        interaction=build_interaction_block(
            check_a=spec_na.check_param,
            check_b=spec_nb.check_param,
            max_far_deviation=float(stats_auto["max_far_deviation"]),
            max_frr_deviation=float(stats_auto["max_frr_deviation"]),
            independent=bool(stats_auto["independent"]),
            within_range_far_deviation=float(stats_auto["within_range_far_deviation"]),
            within_range_frr_deviation=float(stats_auto["within_range_frr_deviation"]),
            extra_pairs=[
                interaction_stats_to_block(
                    stats_fixed,
                    check_a=spec_fe.check_param,
                    check_b=spec_bi.check_param,
                )
            ],
        ),
        runtime_s=runtime_s,
        grid_flags=grid_flags_from_ranges(ranges),
        pairwise_rev=2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "exp6_curves.csv", curve_rows_to_dicts(curves), CURVE_COLUMNS)
    _write_csv(out_dir / "exp6_pairwise.csv", all_pairwise, PAIRWISE_COLUMNS)
    with open(out_dir / "exp6_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_figure:
        _write_sensitivity_figures(curves, ranges, out_dir)

    print(f"[DONE] {out_dir} ({runtime_s:.1f}s)")


if __name__ == "__main__":
    main()
