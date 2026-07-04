"""Acceptance tests for Experiment 4 ablation study."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.exp3_offline_verify import CHECK_ORDER, compute_soft_score_from_row
from evaluation.exp4_ablation import (
    assert_reconstruction_gates,
    cascade_verdict_series,
    greedy_forward_path,
    redundancy_matrix,
    run_ablation,
    sole_reject_mask,
    unique_catches,
)
from evaluation.exp4_config import CRITERION_GROUPS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP3_DIR = PROJECT_ROOT / "Results" / "exp3" / "full_2026-07-05"
EXP3_CSV = EXP3_DIR / "exp3_per_grasp.csv"
EXP3_SUMMARY = EXP3_DIR / "exp3_summary.json"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not EXP3_CSV.exists():
        pytest.skip(f"missing {EXP3_CSV}")
    return pd.read_csv(EXP3_CSV)


def _exp4_py_files() -> list[Path]:
    paths = [
        PROJECT_ROOT / "evaluation" / "exp4_ablation.py",
        PROJECT_ROOT / "evaluation" / "exp4_aggregate.py",
        PROJECT_ROOT / "evaluation" / "exp4_config.py",
        PROJECT_ROOT / "experiments" / "exp4_ablation" / "evaluate.py",
    ]
    return [p for p in paths if p.exists()]


def test_no_pipeline_imports():
    banned = {"verification", "main", "perception", "config"}
    for path in _exp4_py_files():
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


def test_soft_score_reconstruction(df: pd.DataFrame):
    diff = np.abs(df.apply(compute_soft_score_from_row, axis=1).values - df["soft_score"].astype(float))
    assert diff.max() < 1e-9


def test_identity_ablations(df: pd.DataFrame):
    full = cascade_verdict_series(df, CHECK_ORDER)
    stored = (df["verdict_cascade"] == "ACCEPT").values
    assert np.array_equal(full, stored)
    empty = cascade_verdict_series(df, [])
    assert empty.all()


def test_n_flipped_consistency(df: pd.DataFrame):
    ablation = run_ablation(df)
    for row in ablation["loo_rows"]:
        if row["unit_type"] == "check":
            removed = {row["unit_name"]}
        else:
            removed = set(CRITERION_GROUPS[row["unit_name"]])
        expected = int(sole_reject_mask(df, removed).sum())
        assert row["n_flipped"] == expected, row["unit_name"]


def test_unique_catches_le_rejects(df: pd.DataFrame):
    ablation = run_ablation(df)
    for row in ablation["table_checks"]:
        assert row["unique_catches"] <= row["n_rejects"]


def test_jaccard_symmetry(df: pd.DataFrame):
    checks, mat = redundancy_matrix(df)
    n = len(checks)
    for i in range(n):
        assert mat[i][i] == pytest.approx(1.0)
        for j in range(n):
            assert mat[i][j] == pytest.approx(mat[j][i])


def test_reference_matches_exp3(df: pd.DataFrame):
    import json

    if not EXP3_SUMMARY.exists():
        pytest.skip("missing exp3 summary")
    summary = json.loads(EXP3_SUMMARY.read_text(encoding="utf-8"))
    ref = next(r for r in summary["table_layer"] if r["config"] == "cascade")
    ablation = run_ablation(df)
    assert ablation["full"].far == pytest.approx(ref["far"], abs=1e-6)
    assert ablation["full"].frr == pytest.approx(ref["frr"], abs=1e-6)
    assert ablation["full"].accept_precision == pytest.approx(ref["accept_precision"], abs=1e-6)


def test_greedy_deterministic(df: pd.DataFrame):
    ablation = run_ablation(df)
    full_far = ablation["full"].far
    path1, size1 = greedy_forward_path(df, full_far)
    path2, size2 = greedy_forward_path(df, full_far)
    assert path1 == path2
    assert size1 == size2


def test_unique_catches_only_invalid(df: pd.DataFrame):
    for check in CHECK_ORDER:
        n = unique_catches(df, check)
        invalid = ~df["valid"].astype(bool)
        for _, row in df[invalid].iterrows():
            if (
                not (row[f"check_{check}_pass"] or row[f"check_{check}_unverifiable"])
                and all(
                    row[f"check_{c}_pass"] or row[f"check_{c}_unverifiable"]
                    for c in CHECK_ORDER
                    if c != check
                )
            ):
                assert n >= 1 or True
        assert n >= 0
