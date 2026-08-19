#!/usr/bin/env python3
"""Repair Experiment 6 pairwise sweeps without touching OAT outputs.

Recomputes exp6_pairwise.csv and the interaction block only; preserves
exp6_curves.csv, table_ranges, raw_value_mapping, and default_reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.exp6_aggregate import build_interaction_block, interaction_stats_to_block  # noqa: E402
from evaluation.exp6_sensitivity import (  # noqa: E402
    CurveRow,
    FIXED_PAIRWISE,
    interaction_deviation,
    narrowest_pair_specs,
    range_tuple_from_table,
    run_pairwise_sweep,
)
from evaluation.exp6_threshold_map import build_threshold_specs, spec_by_param  # noqa: E402

PAIRWISE_COLUMNS = ["check_a", "tau_a", "check_b", "tau_b", "far", "frr"]
PAIRWISE_REV = 2


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _curves_from_csv(path: Path) -> list[CurveRow]:
    df = pd.read_csv(path)
    return [
        CurveRow(
            check_param=str(r.check_param),
            tau=float(r.tau),
            tau_rel=float(r.tau_rel),
            far=float(r.far),
            frr=float(r.frr),
            accept_rate=float(r.accept_rate),
            accept_precision=float(r.accept_precision),
            n_rejects_by_check=int(r.n_rejects_by_check),
            grid_type="log",
            grid_clamped=False,
        )
        for r in df.itertuples(index=False)
    ]


def _ranges_from_summary(table_ranges: list[dict]) -> list:
    from evaluation.exp6_sensitivity import RangeRow

    return [
        RangeRow(
            check_param=str(r["check_param"]),
            default=float(r["default"]),
            range_lo=float(r["range_lo"]),
            range_hi=float(r["range_hi"]),
            frr_at_half=float(r["frr_at_half"]),
            frr_at_double=float(r["frr_at_double"]),
            far_at_half=float(r["far_at_half"]),
            far_at_double=float(r["far_at_double"]),
            class_=str(r["class"]),
            grid_type=str(r.get("grid_type", "log")),
            grid_clamped=bool(r.get("grid_clamped", False)),
        )
        for r in table_ranges
    ]


def repair_pairwise(
    *,
    exp3_csv: Path,
    run_dir: Path,
    pairwise_rev: int = PAIRWISE_REV,
) -> None:
    summary_path = run_dir / "exp6_summary.json"
    curves_path = run_dir / "exp6_curves.csv"
    pairwise_path = run_dir / "exp6_pairwise.csv"

    if not summary_path.exists() or not curves_path.exists():
        raise SystemExit(f"Missing exp6 outputs in {run_dir}")

    summary_before = json.loads(summary_path.read_text(encoding="utf-8"))
    preserved = {
        "reference": deepcopy(summary_before["reference"]),
        "table_ranges": deepcopy(summary_before["table_ranges"]),
        "default_reproduction": deepcopy(summary_before["meta"]["default_reproduction"]),
        "raw_value_mapping": deepcopy(summary_before["meta"]["raw_value_mapping"]),
    }
    curves_before_text = curves_path.read_text(encoding="utf-8")

    df = pd.read_csv(exp3_csv)
    specs = build_threshold_specs()
    sb = spec_by_param(specs)
    table_ranges = preserved["table_ranges"]
    ranges = _ranges_from_summary(table_ranges)
    curves = _curves_from_csv(curves_path)

    spec_na, spec_nb = narrowest_pair_specs(ranges, specs)
    pairwise_auto = run_pairwise_sweep(df, spec_na, spec_nb)
    stats_auto = interaction_deviation(
        pairwise_auto,
        curves,
        spec_na,
        spec_nb,
        range_a=range_tuple_from_table(table_ranges, spec_na.check_param),
        range_b=range_tuple_from_table(table_ranges, spec_nb.check_param),
    )

    spec_fe, spec_bi = sb[FIXED_PAIRWISE[0]], sb[FIXED_PAIRWISE[1]]
    pairwise_fixed = run_pairwise_sweep(df, spec_fe, spec_bi)
    stats_fixed = interaction_deviation(
        pairwise_fixed,
        curves,
        spec_fe,
        spec_bi,
        range_a=range_tuple_from_table(table_ranges, spec_fe.check_param),
        range_b=range_tuple_from_table(table_ranges, spec_bi.check_param),
    )

    _write_csv(pairwise_path, pairwise_auto + pairwise_fixed, PAIRWISE_COLUMNS)

    summary = deepcopy(summary_before)
    summary["interaction"] = build_interaction_block(
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
    )
    summary["meta"]["pairwise_rev"] = pairwise_rev

    # Regression gates
    assert summary["reference"] == preserved["reference"]
    assert summary["table_ranges"] == preserved["table_ranges"]
    assert summary["meta"]["default_reproduction"] == preserved["default_reproduction"]
    assert summary["meta"]["raw_value_mapping"] == preserved["raw_value_mapping"]
    assert curves_path.read_text(encoding="utf-8") == curves_before_text

    pw = pd.read_csv(pairwise_path)
    assert pw["tau_a"].max() <= 1.0 + 1e-9 or pw["tau_a"].max() <= max(
        s.domain_hi or float("inf") for s in specs if s.domain_hi is not None
    )
    assert float(pw["tau_a"].max()) <= 1.0 + 1e-6
    assert float(pw["tau_b"].max()) <= 1.0 + 1e-6

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[REPAIR] pairwise rows={len(pw)} max_tau_a={pw.tau_a.max():.4f} max_tau_b={pw.tau_b.max():.4f}")
    print(f"[REPAIR] interaction independent={summary['interaction']['independent']}")
    print(f"[REPAIR] within_range far/frr={summary['interaction']['within_range_far_deviation']:.4f}/"
          f"{summary['interaction']['within_range_frr_deviation']:.4f}")
    print(f"[REPAIR] wrote {pairwise_path} and {summary_path} (pairwise_rev={pairwise_rev})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Exp6 pairwise sweeps only")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "Results/exp6/full_2026-07-05",
    )
    parser.add_argument(
        "--exp3-csv",
        type=Path,
        default=PROJECT_ROOT / "Results/exp3/full_2026-07-05/exp3_per_grasp.csv",
    )
    args = parser.parse_args()
    repair_pairwise(exp3_csv=args.exp3_csv.resolve(), run_dir=args.run_dir.resolve())


if __name__ == "__main__":
    main()
