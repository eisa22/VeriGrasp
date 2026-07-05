"""Experiment 5 — offline end-to-end robustness funnel over pipeline + Exp3 results."""

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
from evaluation.exp5_aggregate import build_summary  # noqa: E402
from evaluation.exp5_funnel import (  # noqa: E402
    EXPECTED_TOTALS,
    assert_funnel_gates,
    classify_all_scenes,
)

PER_SCENE_COLUMNS = [
    "scene_id",
    "category_band",
    "funnel_stage",
    "released",
    "released_valid",
    "handover_status_raw",
    "degenerate_plane",
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
                else:
                    out[col] = val
            writer.writerow(out)


def _rows_to_csv_dicts(rows) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        out.append({
            "scene_id": row.scene_id,
            "category_band": row.category_band,
            "funnel_stage": row.funnel_stage,
            "released": row.released if row.released is not None else "",
            "released_valid": row.released_valid if row.released_valid is not None else "",
            "handover_status_raw": row.handover_status_raw,
            "degenerate_plane": row.degenerate_plane,
        })
    return out


def main(argv: list[str] | None = None) -> None:
    cfg = _load_eval_config()
    parser = argparse.ArgumentParser(description="Experiment 5 robustness funnel")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / cfg.get("data_root", "Data/blender_dataset"),
    )
    parser.add_argument(
        "--exp3-dir",
        type=Path,
        default=PROJECT_ROOT / cfg.get("exp3_dir", "Results/exp3/full_2026-07-05"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / cfg.get("out_dir", "Results/exp5/full_2026-07-05"),
    )
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    exp3_dir = args.exp3_dir.resolve()
    out_dir = args.out_dir.resolve()

    csv_path = exp3_dir / "exp3_per_grasp.csv"
    summary_path = exp3_dir / "exp3_summary.json"
    if not csv_path.exists():
        raise SystemExit(f"Missing input: {csv_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing input: {summary_path}")

    soft_classes = set(cfg.get("soft_classes", []))
    band_order = list(cfg.get("category_rows", []))

    t0 = time.perf_counter()
    exp3_df = pd.read_csv(csv_path)
    with open(summary_path, encoding="utf-8") as f:
        exp3_summary = json.load(f)

    print(f"[EXP5] Classifying {EXPECTED_TOTALS['n_scenes']} scenes …")
    rows = classify_all_scenes(data_root, exp3_df, soft_classes=soft_classes)

    gate_stats = assert_funnel_gates(rows, exp3_df)
    print(
        f"[GATE] OK stages={gate_stats['stage_counts']} "
        f"inferred={gate_stats['n_inferred_handover']}"
    )

    ref_layer = next(r for r in exp3_summary["table_layer"] if r["config"] == "cascade")
    if abs(gate_stats["accept_rate"] - ref_layer["accept_rate"]) > 1e-6:
        raise SystemExit(
            f"accept_rate mismatch: {gate_stats['accept_rate']} vs {ref_layer['accept_rate']}"
        )
    if abs(gate_stats["accept_precision"] - ref_layer["accept_precision"]) > 1e-6:
        raise SystemExit(
            f"accept_precision mismatch: {gate_stats['accept_precision']} "
            f"vs {ref_layer['accept_precision']}"
        )

    runtime_s = time.perf_counter() - t0
    summary = build_summary(
        exp3_run_id=exp3_dir.name,
        git_commit=_git_commit(),
        rows=rows,
        gate_stats=gate_stats,
        band_order=band_order,
        runtime_s=runtime_s,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "exp5_per_scene.csv", _rows_to_csv_dicts(rows), PER_SCENE_COLUMNS)
    with open(out_dir / "exp5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] {out_dir} ({runtime_s:.1f}s)")


if __name__ == "__main__":
    main()
