"""Acceptance tests for Experiment 6 threshold sensitivity."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.exp6_sensitivity import (
    FIXED_PAIRWISE,
    REFERENCE_FAR,
    REFERENCE_FRR,
    assert_default_reproduction,
    assert_grids_contain_default,
    assert_monotonicity,
    run_oat_sweeps,
    run_pairwise_sweep,
)
from evaluation.exp6_threshold_map import build_pairwise_grid, build_threshold_specs, validate_round_trip
from evaluation.exp4_ablation import assert_reconstruction_gates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP3_DIR = PROJECT_ROOT / "Results" / "exp3" / "full_2026-07-05"
EXP3_CSV = EXP3_DIR / "exp3_per_grasp.csv"
EXP6_DIR = PROJECT_ROOT / "Results" / "exp6" / "full_2026-07-05"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not EXP3_CSV.exists():
        pytest.skip(f"missing {EXP3_CSV}")
    frame = pd.read_csv(EXP3_CSV)
    if "n_blocking_points" not in frame.columns:
        pytest.skip("run scripts/patch_exp3_n_blocking.py first")
    return frame


@pytest.fixture(scope="module")
def specs():
    return build_threshold_specs()


def _exp6_py_files() -> list[Path]:
    paths = [
        PROJECT_ROOT / "evaluation" / "exp6_threshold_map.py",
        PROJECT_ROOT / "evaluation" / "exp6_sensitivity.py",
        PROJECT_ROOT / "evaluation" / "exp6_aggregate.py",
        PROJECT_ROOT / "experiments" / "exp6_sensitivity" / "evaluate.py",
    ]
    return [p for p in paths if p.exists()]


def test_no_pipeline_imports():
    banned = {"verification", "main", "perception", "config"}
    for path in _exp6_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in banned, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in banned, f"{path.name} imports from {node.module}"


def test_cascade_reconstruction(df: pd.DataFrame):
    stats = assert_reconstruction_gates(df)
    assert stats["cascade_mismatch"] == 0
    assert stats["n_checked"] == 616


def test_round_trip_default(df: pd.DataFrame, specs):
    mismatches = validate_round_trip(df, specs)
    assert all(v == 0 for v in mismatches.values()), mismatches


def test_default_reproduction(df: pd.DataFrame, specs):
    _, _, ref = run_oat_sweeps(df, specs)
    assert_default_reproduction(ref)
    assert ref.far == pytest.approx(REFERENCE_FAR)
    assert ref.frr == pytest.approx(REFERENCE_FRR)


def test_monotonicity(df: pd.DataFrame, specs):
    curves, _, _ = run_oat_sweeps(df, specs)
    for spec in specs:
        assert_monotonicity(curves, spec)


def test_grids_contain_default(df: pd.DataFrame, specs):
    curves, _, _ = run_oat_sweeps(df, specs)
    assert_grids_contain_default(curves, specs)


def test_corridor_integer_grid_at_default(df: pd.DataFrame, specs):
    default_tol = 5
    recomputed = df["n_blocking_points"].astype(int) <= default_tol
    stored = df["check_corridor_clear_pass"].astype(bool)
    assert (recomputed == stored).all()
    assert int((~stored).sum()) == 92


def test_pairwise_grid_clamped_to_domain(specs):
    sb = {s.check_param: s for s in specs}
    sa = sb["suction_area.min_area_ratio"]
    sb_in = sb["bbox_inlier.inlier_min"]
    ga = build_pairwise_grid(sa)
    gb = build_pairwise_grid(sb_in)
    assert ga.max() <= 1.0 + 1e-15
    assert gb.max() <= 1.0 + 1e-15
    assert sa.default in ga
    assert sb_in.default in gb


def test_pairwise_csv_has_bbox_pair(df: pd.DataFrame, specs):
    from evaluation.exp6_threshold_map import spec_by_param

    sb = spec_by_param(specs)
    rows = run_pairwise_sweep(df, sb[FIXED_PAIRWISE[0]], sb[FIXED_PAIRWISE[1]])
    assert len(rows) <= 49
    assert max(r["tau_a"] for r in rows) <= 1.0 + 1e-9
    assert max(r["tau_b"] for r in rows) <= 1.0 + 1e-9


@pytest.mark.skipif(not (EXP6_DIR / "exp6_summary.json").exists(), reason="exp6 not run")
def test_outputs_exist():
    assert (EXP6_DIR / "exp6_curves.csv").exists()
    assert (EXP6_DIR / "exp6_pairwise.csv").exists()
    summary = json.loads((EXP6_DIR / "exp6_summary.json").read_text(encoding="utf-8"))
    assert summary["reference"]["far"] == REFERENCE_FAR
    assert len(summary["table_ranges"]) == 20
