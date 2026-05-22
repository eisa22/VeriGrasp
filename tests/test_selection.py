"""Unit tests for Stage 10 target selection."""

from __future__ import annotations

import numpy as np

from perception.candidate import BottomInference, CandidateOut
from perception.selection import select_target_smallest_z


def _make_candidate(
    cid: str,
    height_m: float,
    *,
    top_z: float | None = None,
    center_xy: tuple[float, float] = (0.0, 0.0),
    obb_extent_xy: float = 0.20,
    confidence: float = 0.9,
    method: str = "from_neighbor",
) -> CandidateOut:
    pts = np.zeros((4, 3))
    top = top_z if top_z is not None else 0.5 + height_m
    return CandidateOut(
        candidate_id=cid,
        mask_2d=np.zeros((10, 10), dtype=np.uint8),
        points_3d=pts,
        centroid_3d=np.zeros(3),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        surface_area_m2=0.01,
        top_surface_height=top,
        bbox_2d=(0, 0, 10, 10),
        debug={
            "label": f"L_{cid}",
            "center_xy": list(center_xy),
            "obb_extent_xy": obb_extent_xy,
        },
        bottom=BottomInference(
            bottom_z=top - height_m,
            bottom_method=method,
            bottom_confidence=confidence,
            bottom_residual_m=0.0,
            used_neighbor_ids=[],
            height_m=height_m,
            parcel_obb={
                "center": [0.0, 0.0, 0.0],
                "extents": [0.1, 0.1, height_m],
                "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "corners_3d": [[0, 0, 0]] * 8,
            },
        ),
    )


def test_within_top_band_smallest_extent_wins():
    """Same layer (within 5 cm): smallest bounding-box height wins."""
    cs = [
        _make_candidate("a", height_m=0.40, top_z=0.60),
        _make_candidate("b", height_m=0.12, top_z=0.58),
        _make_candidate("c", height_m=0.25, top_z=0.59),
    ]
    res = select_target_smallest_z(cs, top_band_m=0.05)
    assert res.primary.candidate.candidate_id == "b"
    assert res.primary.score == 0.12
    assert res.primary.reason == "highest_tier_smallest_extent_fewest_neighbors"
    assert [t.candidate.candidate_id for t in res.ranking] == ["b", "c", "a"]
    assert res.max_top_m == 0.60


def test_lower_layer_rejected_even_if_smallest_box():
    """A small box on a lower layer must not beat the top layer."""
    cs = [
        _make_candidate("high_big", height_m=0.50, top_z=0.80),
        _make_candidate("low_tiny", height_m=0.08, top_z=0.30),
    ]
    res = select_target_smallest_z(cs, top_band_m=0.05)
    assert res.primary.candidate.candidate_id == "high_big"
    assert len(res.ranking) == 1
    assert any("below_top_tier" in r for _, r in res.rejected)


def test_top_band_5cm_includes_edge_case():
    """Parcel exactly 5 cm below max top is still in the tier."""
    cs = [
        _make_candidate("top", height_m=0.30, top_z=0.70),
        _make_candidate("edge", height_m=0.10, top_z=0.65),
        _make_candidate("below", height_m=0.05, top_z=0.64),
    ]
    res = select_target_smallest_z(cs, top_band_m=0.05)
    assert res.primary.candidate.candidate_id == "edge"
    ids_ranked = [t.candidate.candidate_id for t in res.ranking]
    assert ids_ranked == ["edge", "top"]
    assert any(c.candidate_id == "below" for c, _ in res.rejected)


def test_low_confidence_is_rejected():
    cs = [
        _make_candidate("a", height_m=0.05, top_z=0.60, confidence=0.30),
        _make_candidate("b", height_m=0.20, top_z=0.59, confidence=0.80),
    ]
    res = select_target_smallest_z(cs, min_confidence=0.5)
    assert res.primary.candidate.candidate_id == "b"
    assert len(res.rejected) == 1
    assert "low_confidence" in res.rejected[0][1]


def test_method_filter_restricts_eligible():
    cs = [
        _make_candidate("a", height_m=0.05, top_z=0.60, method="from_pallet"),
        _make_candidate("b", height_m=0.20, top_z=0.59, method="from_neighbor"),
    ]
    res = select_target_smallest_z(cs, require_method=("from_neighbor",))
    assert res.primary.candidate.candidate_id == "b"
    assert any("method" in r for _, r in res.rejected)


def test_no_eligible_candidates_returns_none_primary():
    cs = [_make_candidate("a", height_m=0.1, confidence=0.1)]
    res = select_target_smallest_z(cs, min_confidence=0.5)
    assert res.primary is None
    assert res.ranking == []
    assert len(res.rejected) == 1


def test_fewest_neighbors_breaks_height_tie():
    """Same top tier and same height: parcel with fewer lateral peers wins."""
    cs = [
        _make_candidate("inner", height_m=0.20, top_z=0.60, center_xy=(0.0, 0.0)),
        _make_candidate("between", height_m=0.20, top_z=0.59, center_xy=(0.30, 0.0)),
        _make_candidate("edge", height_m=0.20, top_z=0.58, center_xy=(1.00, 0.0)),
    ]
    res = select_target_smallest_z(cs, neighbor_radius_m=0.30)
    assert res.primary.candidate.candidate_id == "edge"
    assert res.primary.n_lateral_peers == 0
    assert res.ranking[1].candidate.candidate_id == "between"
    assert res.ranking[1].n_lateral_peers == 1


def test_serializable_handover_includes_policy():
    cs = [
        _make_candidate("a", height_m=0.40, top_z=0.60),
        _make_candidate("b", height_m=0.10, top_z=0.58),
    ]
    res = select_target_smallest_z(cs)
    out = res.to_serializable()
    assert out["selection_policy"] == "highest_tier_smallest_extent_fewest_neighbors"
    assert out["primary"]["candidate"]["candidate_id"] == "b"
    assert out["primary"]["n_lateral_peers"] >= 0
    assert out["top_band_m"] == 0.05
    assert out["max_top_m"] == 0.60
