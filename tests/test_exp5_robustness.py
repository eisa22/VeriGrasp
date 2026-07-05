"""Acceptance tests for Experiment 5 robustness funnel."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.exp5_funnel import (
    EXPECTED_TOTALS,
    STATUS_MAPPING,
    assert_funnel_gates,
    classify_all_scenes,
    funnel_stage_from_raw,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP3_DIR = PROJECT_ROOT / "Results" / "exp3" / "full_2026-07-05"
EXP3_CSV = EXP3_DIR / "exp3_per_grasp.csv"
EXP3_SUMMARY = EXP3_DIR / "exp3_summary.json"
DATA_ROOT = PROJECT_ROOT / "Data" / "blender_dataset"


@pytest.fixture(scope="module")
def exp3_df() -> pd.DataFrame:
    if not EXP3_CSV.exists():
        pytest.skip(f"missing {EXP3_CSV}")
    return pd.read_csv(EXP3_CSV)


@pytest.fixture(scope="module")
def funnel_rows(exp3_df: pd.DataFrame):
    if not DATA_ROOT.exists():
        pytest.skip(f"missing {DATA_ROOT}")
    soft = {
        "Burlap_Sack",
        "Food_Packaging_Bag",
        "Paper_Coffee_Bag",
        "Paper_Shopping_Bag",
        "Small_Pouch",
        "Small_Shipping_Bag",
        "Space_Food_Bag",
        "Sachets_Package",
        "Envelope_Stack",
        "Vintage_Envelope",
        "Asset_Deformed_Package",
    }
    return classify_all_scenes(DATA_ROOT, exp3_df, soft_classes=soft)


def _exp5_py_files() -> list[Path]:
    paths = [
        PROJECT_ROOT / "evaluation" / "exp5_funnel.py",
        PROJECT_ROOT / "evaluation" / "exp5_aggregate.py",
        PROJECT_ROOT / "experiments" / "exp5_robustness" / "evaluate.py",
    ]
    return [p for p in paths if p.exists()]


def test_no_pipeline_imports():
    banned = {"verification", "main", "perception", "config"}
    for path in _exp5_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in banned, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in banned, f"{path.name} imports from {node.module}"


def test_status_mapping_complete():
    for raw, stage in STATUS_MAPPING.items():
        assert funnel_stage_from_raw(raw) == stage


def test_funnel_exhaustive(funnel_rows):
    assert len(funnel_rows) == EXPECTED_TOTALS["n_scenes"]
    assert len({r.scene_id for r in funnel_rows}) == EXPECTED_TOTALS["n_scenes"]
    stages = [r.funnel_stage for r in funnel_rows]
    assert sum(stages.count(s) for s in ("no_candidates", "no_target", "rejected", "released")) == 728


def test_hard_expected_totals(funnel_rows, exp3_df: pd.DataFrame):
    stats = assert_funnel_gates(funnel_rows, exp3_df)
    assert stats["n_grasp_scenes"] == 616
    assert stats["released"] == 389
    assert stats["released_valid"] == 265
    assert stats["released_invalid"] == 124
    assert stats["rejected"] == 227
    assert stats["pre_grasp"] == 112


def test_exp3_consistency(funnel_rows, exp3_df: pd.DataFrame):
    stats = assert_funnel_gates(funnel_rows, exp3_df)
    if not EXP3_SUMMARY.exists():
        pytest.skip(f"missing {EXP3_SUMMARY}")
    summary = json.loads(EXP3_SUMMARY.read_text(encoding="utf-8"))
    cascade = next(r for r in summary["table_layer"] if r["config"] == "cascade")
    assert abs(stats["accept_rate"] - cascade["accept_rate"]) < 1e-6
    assert abs(stats["accept_precision"] - cascade["accept_precision"]) < 1e-6


def test_status_mapping_observed(funnel_rows):
    raw_seen = {r.handover_status_raw for r in funnel_rows}
    unknown = raw_seen - set(STATUS_MAPPING)
    assert not unknown, f"unmapped raw statuses: {sorted(unknown)}"
