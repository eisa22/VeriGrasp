"""Experiment 3: metric sanity and summary consistency tests."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from evaluation.exp3_aggregate import build_summary
from evaluation.exp3_metrics import (
    auc_trapezoid,
    binary_rates,
    layer_metrics,
    per_band_layer,
    roc_curve,
)
from evaluation.exp3_offline_verify import compute_soft_score_from_row

OLD_RUN = Path(__file__).resolve().parents[1] / "Results/exp3/full_2026-07-04/exp3_per_grasp.csv"


def _synthetic_rows(n: int = 200, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    bands = ["baseline", "mixed", "dense"]
    for i in range(n):
        valid = bool(rng.random() > 0.3)
        cascade = bool(rng.random() > 0.5)
        rows.append({
            "scene_id": f"scene_{i:03d}",
            "category_band": bands[i % len(bands)],
            "target_matched": True,
            "gt_class": "Box",
            "valid": valid,
            "violated_criteria": "" if valid else "surface_contact",
            "verdict_cascade": "ACCEPT" if cascade else "REJECT",
            "decisive_check": "",
            "soft_score": float(rng.normal()),
            "check_planarity_pass": True,
            "check_planarity_margin": float(rng.normal()),
            "check_planarity_unverifiable": False,
        })
    return rows


def test_baseline_accept_precision_equals_base_rate():
    rows = _synthetic_rows(100)
    table, _, stats = layer_metrics(rows)
    baseline = next(r for r in table if r["config"] == "baseline")
    assert baseline["accept_precision"] == pytest.approx(stats["base_rate"], abs=1e-12)
    assert baseline["far"] == pytest.approx(1.0)
    assert baseline["frr"] == pytest.approx(0.0)


def test_per_band_n_sums_to_total():
    rows = _synthetic_rows(120)
    bands = ["baseline", "mixed", "dense"]
    pb = per_band_layer(rows, bands)
    total_row = next(r for r in pb if r["band"] == "total")
    assert total_row["n"] == len(rows)
    assert sum(r["n"] for r in pb if r["band"] != "total") == len(rows)


def test_random_margin_auc_near_half():
    rng = np.random.default_rng(42)
    n = 500
    valid = rng.random(n) > 0.5
    margins = rng.normal(size=n)
    roc = roc_curve(margins, valid, higher_is_pass=True)
    auc = auc_trapezoid(roc["fpr"], roc["tpr"])
    assert auc is not None
    assert auc == pytest.approx(0.5, abs=0.08)


def test_cascade_point_on_soft_score_roc():
    """Cascade operating point should be plottable against the soft-score ROC."""
    rows = _synthetic_rows(300, seed=7)
    _, roc, _ = layer_metrics(rows)
    cp = roc["cascade_point"]
    assert 0.0 <= cp["fpr"] <= 1.0
    assert 0.0 <= cp["tpr"] <= 1.0
    assert len(roc["fpr"]) >= 2
    assert roc["auc"] is not None


def test_build_summary_schema_keys():
    rows = _synthetic_rows(50)
    meta = {
        "run_id": "test",
        "config_hash": "abc",
        "git_commit": "deadbeef",
        "visibility_rule": "relative_median",
        "n_scenes": 50,
        "n_no_target": 0,
        "n_grasps": len(rows),
        "n_target_unmatched": 2,
        "oracle_params": {
            "tol_contain_mm": 5,
            "tol_contact_mm": 20,
            "tol_normal_deg": 15,
        },
    }
    summary = build_summary(rows, ["baseline", "mixed", "dense"], meta)
    assert "table_checks" in summary
    assert "table_layer" in summary
    assert "roc_soft_score" in summary
    assert "base_rate" in summary
    assert "total" in summary["base_rate"]
    assert "per_band" in summary["base_rate"]


def test_binary_rates_far_frr():
    valid = np.array([True, True, False, False])
    accept = np.array([True, False, True, False])
    br = binary_rates(accept, valid)
    assert br.frr == pytest.approx(0.5)  # 1 rejected valid / 2 valid
    assert br.far == pytest.approx(0.5)  # 1 accepted invalid / 2 invalid


@pytest.mark.skipif(not OLD_RUN.exists(), reason="prior exp3 run not on disk")
def test_soft_score_reconstruction_matches_prior_scored_rows():
    """Rows that already had a cascade soft_score must match margin reconstruction."""
    rows = list(csv.DictReader(OLD_RUN.open(encoding="utf-8")))
    scored = [r for r in rows if str(r.get("soft_score", "")).strip() != ""]
    assert len(scored) >= 400
    mismatches = []
    for r in scored:
        old = float(r["soft_score"])
        new = compute_soft_score_from_row(r)
        if abs(old - new) > 1e-9:
            mismatches.append((r["scene_id"], old, new))
    assert not mismatches, mismatches[:5]
