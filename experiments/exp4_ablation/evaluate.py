"""Experiment 4 — offline ablation over persisted Experiment 3 results."""

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
from evaluation.exp4_ablation import assert_reconstruction_gates, run_ablation  # noqa: E402
from evaluation.exp4_aggregate import build_summary  # noqa: E402

LOO_COLUMNS = [
    "unit_type",
    "unit_name",
    "n_flipped",
    "delta_far",
    "delta_frr",
    "delta_accept_precision",
    "far",
    "frr",
    "accept_rate",
    "accept_precision",
    "delta_soft_auc",
]


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
            out = {}
            for col in columns:
                val = row.get(col, "")
                if val == "" or val is None:
                    out[col] = ""
                elif col == "delta_soft_auc" and row.get("unit_type") == "criterion":
                    out[col] = ""
                else:
                    out[col] = val
            writer.writerow(out)


def _write_greedy_figure(greedy_path: list[dict], out_path: Path) -> None:
    if not greedy_path:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [p["step"] for p in greedy_path]
    far = [p["far"] for p in greedy_path]
    frr = [p["frr"] for p in greedy_path]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, far, marker="o", linewidth=1.2, label="FAR")
    ax.plot(steps, frr, marker="s", linewidth=1.2, label="FRR")
    ax.set_xlabel("greedy step")
    ax.set_ylabel("rate")
    ax.set_title("Experiment 4 — greedy forward selection")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    cfg = _load_eval_config()
    parser = argparse.ArgumentParser(description="Experiment 4 ablation study")
    parser.add_argument(
        "--exp3-dir",
        type=Path,
        default=PROJECT_ROOT / cfg.get("exp3_dir", "Results/exp3/full_2026-07-05"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / cfg.get("out_dir", "Results/exp4/full_2026-07-05"),
    )
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    exp3_dir = args.exp3_dir.resolve()
    out_dir = args.out_dir.resolve()
    csv_path = exp3_dir / "exp3_per_grasp.csv"
    summary_path = exp3_dir / "exp3_summary.json"
    if not csv_path.exists():
        raise SystemExit(f"Missing input: {csv_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing input: {summary_path}")

    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    with open(summary_path, encoding="utf-8") as f:
        exp3_summary = json.load(f)

    print(f"[GATE] Cascade + soft-score reconstruction on {len(df)} grasps …")
    reconstruction = assert_reconstruction_gates(df)
    print(
        f"[GATE] OK cascade mismatches={reconstruction['cascade_mismatch']} "
        f"soft_max_diff={reconstruction['soft_max_diff']:.2e}"
    )

    ablation = run_ablation(df)

    if abs(ablation["identity_all_removed_far"] - 1.0) > 1e-9:
        raise SystemExit("Identity check failed: all-removed FAR != 1.0")
    if abs(ablation["identity_all_removed_frr"]) > 1e-9:
        raise SystemExit("Identity check failed: all-removed FRR != 0.0")
    if abs(ablation["identity_all_removed_accept_rate"] - 1.0) > 1e-9:
        raise SystemExit("Identity check failed: all-removed accept_rate != 1.0")

    ref_layer = next(r for r in exp3_summary["table_layer"] if r["config"] == "cascade")
    reference = {
        "far": ref_layer["far"],
        "frr": ref_layer["frr"],
        "accept_precision": ref_layer["accept_precision"],
        "base_rate": exp3_summary["base_rate"]["total"],
    }
    if abs(ablation["full"].far - reference["far"]) > 1e-6:
        raise SystemExit(
            f"Full-layer FAR mismatch vs exp3 summary: {ablation['full'].far} vs {reference['far']}"
        )

    runtime_s = time.perf_counter() - t0
    summary = build_summary(
        exp3_run_id=exp3_dir.name,
        git_commit=_git_commit(),
        n_grasps=len(df),
        reconstruction_check=reconstruction,
        reference=reference,
        ablation=ablation,
        runtime_s=runtime_s,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "exp4_loo.csv", ablation["loo_rows"], LOO_COLUMNS)
    with open(out_dir / "exp4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_figure:
        fig_path = out_dir / "figures" / "exp4_greedy_path.pdf"
        _write_greedy_figure(ablation["greedy_path"], fig_path)
        print(f"[FIGURE] geschrieben -> {fig_path}")

    print(f"[DONE] {out_dir} ({runtime_s:.1f}s)")


if __name__ == "__main__":
    main()
