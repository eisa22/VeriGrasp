"""Experiment 2: units, sign conventions, cue mapping, aggregation."""

import numpy as np
import pytest

from evaluation.exp2_aggregate import (
    build_summary,
    table_bottom,
    table_centroid,
    visibility_strata,
)
from evaluation.exp2_geometry import PredCandidateGeometry
from evaluation.exp2_gt import GtObjectGeometry
from evaluation.exp2_metrics import bottom_cue, candidate_errors


def _gt(center_xy=(0.0, 0.0), h_top=0.2, h_bottom=0.0, yaw=0.0,
        long_m=0.3, short_m=0.2):
    return GtObjectGeometry(
        instance_id=0, class_name="Box", visible_pixels=100,
        center_camera=np.zeros(3), center_xy=np.asarray(center_xy, dtype=np.float64),
        h_top=h_top, h_bottom=h_bottom,
        top_normal=np.array([0.0, 0.0, -1.0]),
        footprint_yaw_deg=yaw, footprint_long_m=long_m, footprint_short_m=short_m,
        footprint_aspect=long_m / short_m, z_top_mean=2.3,
    )


def _pred(centroid_xy=(0.0, 0.0), h_top=0.2, h_bottom=0.0, yaw=0.0,
          long_m=0.3, short_m=0.2, method="measured", source=None):
    return PredCandidateGeometry(
        candidate_id="c0",
        centroid_xy=np.asarray(centroid_xy, dtype=np.float64),
        h_top=h_top, h_bottom=h_bottom,
        footprint_yaw_deg=yaw, footprint_long_m=long_m, footprint_short_m=short_m,
        height_m=h_top - h_bottom, bottom_method=method,
        bottom_confidence=0.9, neighbor_source=source,
    )


def test_e_lat_in_millimetres():
    err = candidate_errors(_pred(centroid_xy=(0.03, 0.04)), _gt())
    assert err.e_lat_mm == pytest.approx(50.0)  # 3-4-5 triangle, 0.05 m


def test_e_top_sign_positive_when_pipeline_too_high():
    err = candidate_errors(_pred(h_top=0.25), _gt(h_top=0.2))
    assert err.e_top_mm_signed == pytest.approx(50.0)
    err = candidate_errors(_pred(h_top=0.15), _gt(h_top=0.2))
    assert err.e_top_mm_signed == pytest.approx(-50.0)


def test_e_bottom_signed():
    err = candidate_errors(_pred(h_bottom=0.02), _gt(h_bottom=0.0))
    assert err.e_bottom_mm_signed == pytest.approx(20.0)


def test_extent_relative_errors():
    err = candidate_errors(
        _pred(long_m=0.33, short_m=0.18, h_top=0.24, h_bottom=0.0),
        _gt(long_m=0.3, short_m=0.2, h_top=0.2, h_bottom=0.0),
    )
    assert err.ext_err_long_rel == pytest.approx(0.1)
    assert err.ext_err_short_rel == pytest.approx(-0.1)
    assert err.ext_err_height_rel == pytest.approx(0.2)


def test_bottom_cue_mapping_priorities():
    assert bottom_cue("from_neighbor", "match") == ("overlap_matched_parcel", 1)
    assert bottom_cue("from_neighbor", "overlap") == ("overlap_candidate", 2)
    assert bottom_cue("from_neighbor", "lateral") == ("lateral_neighbor", 3)
    assert bottom_cue("from_neighbor", "gradient_global") == ("gradient_global", 4)
    assert bottom_cue("from_neighbor", "gradient") == ("gradient_ring", 5)
    assert bottom_cue("from_neighbor", "histogram") == ("histogram_ring", 6)
    assert bottom_cue("from_neighbor", "scene_plane") == ("scene_plane", 7)
    assert bottom_cue("measured", None)[0] == "measured_visible"
    assert bottom_cue("from_pallet", None)[0] == "fallback_pallet"
    assert bottom_cue("uncertain", None)[0] == "fallback_uncertain"


def _row(band="baseline", packaging="rigid", e_lat=10.0, e_top=5.0,
         e_bottom=2.0, cue="measured_visible", vis=0.95):
    return {
        "scene_id": "scene_000", "category_band": band, "gt_instance_id": 0,
        "gt_class": "Box", "packaging_type": packaging,
        "visible_px": 100, "visibility_ratio": vis, "match_iou": 0.9,
        "e_lat_mm": e_lat, "e_top_mm_signed": e_top,
        "ext_err_long_rel": 0.1, "ext_err_short_rel": -0.1,
        "ext_err_height_rel": 0.05, "yaw_err_deg": 2.0, "yaw_fold_deg": 180,
        "e_bottom_mm_signed": e_bottom, "bottom_cue": cue,
        "bottom_confidence": 0.9, "is_primary_target": False,
    }


def test_table_centroid_soft_excluded_from_bands():
    rows = [
        _row(band="baseline", packaging="rigid", e_lat=10.0),
        _row(band="baseline", packaging="soft", e_lat=100.0),
    ]
    table = table_centroid(rows, ["baseline"])
    baseline = next(t for t in table if t["band"] == "baseline")
    soft = next(t for t in table if t["band"] == "soft")
    total = next(t for t in table if t["band"] == "total")
    assert baseline["n"] == 1 and baseline["med_e_lat_mm"] == pytest.approx(10.0)
    assert soft["n"] == 1 and soft["med_e_lat_mm"] == pytest.approx(100.0)
    assert total["n"] == 1  # total pools rigid only


def test_rate_lat_30():
    rows = [_row(e_lat=v) for v in (10.0, 25.0, 40.0, 60.0)]
    table = table_centroid(rows, ["baseline"])
    total = next(t for t in table if t["band"] == "total")
    assert total["rate_lat_30"] == pytest.approx(0.5)


def test_signed_median_and_abs_p95_for_e_top():
    rows = [_row(e_top=v) for v in (-30.0, -10.0, 5.0)]
    table = table_centroid(rows, ["baseline"])
    total = next(t for t in table if t["band"] == "total")
    assert total["med_e_top_mm"] == pytest.approx(-10.0)  # median of signed
    assert total["p95_abs_e_top_mm"] == pytest.approx(28.0)  # p95 of |values|


def test_table_bottom_grouped_by_cue():
    rows = [
        _row(cue="overlap_matched_parcel", e_bottom=1.0),
        _row(cue="overlap_matched_parcel", e_bottom=3.0),
        _row(cue="fallback_pallet", e_bottom=-5.0),
    ]
    table = table_bottom(rows)
    top = next(t for t in table if t["cue"] == "overlap_matched_parcel")
    assert top["priority"] == 1 and top["n"] == 2
    assert top["med_e_bottom_mm"] == pytest.approx(2.0)
    total = next(t for t in table if t["cue"] == "total")
    assert total["n"] == 3


def test_visibility_strata_partition():
    rows = [_row(vis=0.95), _row(vis=0.7), _row(vis=0.3), _row(vis=0.9)]
    strata = visibility_strata(rows)
    by_label = {s["stratum"]: s["n"] for s in strata}
    assert by_label == {">=0.9": 2, "0.5-0.9": 1, "<0.5": 1}


def test_build_summary_meta_counts():
    cand = [_row()]
    grasps = [
        {"scene_id": "scene_000", "category_band": "baseline", "gt_class": "Box",
         "theta_deg": 5.0, "within_12deg": True, "within_30deg": True,
         "status": "evaluated"},
        {"scene_id": "scene_001", "category_band": "baseline", "gt_class": "",
         "theta_deg": "", "within_12deg": "", "within_30deg": "",
         "status": "target_unmatched"},
        {"scene_id": "scene_002", "category_band": "baseline", "gt_class": "",
         "theta_deg": "", "within_12deg": "", "within_30deg": "",
         "status": "no_target"},
    ]
    summary = build_summary(cand, grasps, ["baseline"], {"n_scenes": 3})
    assert summary["meta"]["n_matched_candidates"] == 1
    assert summary["meta"]["n_grasp_scenes"] == 1
    assert summary["meta"]["n_target_unmatched"] == 1
    assert {t["band"] for t in summary["table_normal"]} == {"baseline", "total"}
