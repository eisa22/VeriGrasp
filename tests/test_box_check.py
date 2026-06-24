"""Tests for the deterministic, visibility-aware OBB verification (Stage 1).

Synthetic top-down cuboids exercise PASS / FAIL / UNVERIFIABLE plus the
reproducibility guarantee (identical results across runs).
"""

from __future__ import annotations

import numpy as np

from verification.box_check import (
    FAIL,
    PASS,
    UNVERIFIABLE,
    verify_box_placement,
)

# Camera at origin looking down +z; box centred at z=1.0 so its visible top
# face (toward the camera) sits at z = center_z - half_z.
_CENTER = np.array([0.0, 0.0, 1.0])
_R = np.eye(3)


def _top_face_cloud(extents, nx=60, ny=40, jitter=0.0, seed=0):
    """Dense planar sampling of the box top face (the sensor-visible face)."""
    half = np.asarray(extents, dtype=np.float64) * 0.5
    xs = np.linspace(-half[0], half[0], nx)
    ys = np.linspace(-half[1], half[1], ny)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack(
        [
            gx.ravel() + _CENTER[0],
            gy.ravel() + _CENTER[1],
            np.full(gx.size, _CENTER[2] - half[2]),
        ],
        axis=1,
    )
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        pts = pts + rng.normal(0.0, jitter, size=pts.shape)
    return pts


def test_correct_box_passes():
    extents = np.array([0.30, 0.20, 0.25])
    pts = _top_face_cloud(extents)
    res = verify_box_placement(pts, _CENTER, _R, extents)
    assert res.verdict == PASS, res.reasons
    # Horizontal axes are observable from the top face; vertical is not.
    assert res.extent_observed[0] and res.extent_observed[1]
    assert not res.extent_observed[2]


def test_oversized_box_fails_on_extent():
    extents_true = np.array([0.30, 0.20, 0.25])
    pts = _top_face_cloud(extents_true)
    # Claimed box is far larger in x than the supporting points.
    extents_claimed = np.array([0.50, 0.20, 0.25])
    res = verify_box_placement(pts, _CENTER, _R, extents_claimed)
    assert res.verdict == FAIL
    assert any("extent axis x" in r and "rel_dev" in r for r in res.reasons)


def test_undersized_box_fails_on_extent():
    extents_true = np.array([0.40, 0.30, 0.25])
    pts = _top_face_cloud(extents_true)
    # Claimed box much smaller than the data span -> points spill outside.
    extents_claimed = np.array([0.20, 0.30, 0.25])
    res = verify_box_placement(pts, _CENTER, _R, extents_claimed)
    assert res.verdict == FAIL


def test_too_few_points_unverifiable():
    extents = np.array([0.30, 0.20, 0.25])
    pts = _top_face_cloud(extents)[:50]
    res = verify_box_placement(pts, _CENTER, _R, extents)
    assert res.verdict == UNVERIFIABLE
    assert any("too few" in r for r in res.reasons)


def test_gappy_face_unverifiable():
    extents = np.array([0.30, 0.20, 0.25])
    # Enough points overall, but all crammed in one corner -> low coverage.
    half = extents * 0.5
    rng = np.random.default_rng(1)
    corner = np.stack(
        [
            rng.uniform(-half[0], -half[0] + 0.02, 400) + _CENTER[0],
            rng.uniform(-half[1], -half[1] + 0.02, 400) + _CENTER[1],
            np.full(400, _CENTER[2] - half[2]),
        ],
        axis=1,
    )
    res = verify_box_placement(corner, _CENTER, _R, extents)
    assert res.verdict == UNVERIFIABLE
    assert any("min_coverage" in r for r in res.reasons)


def test_determinism():
    extents = np.array([0.30, 0.20, 0.25])
    pts = _top_face_cloud(extents, jitter=0.002, seed=7)
    a = verify_box_placement(pts, _CENTER, _R, extents)
    b = verify_box_placement(pts, _CENTER, _R, extents)
    assert a.to_serializable() == b.to_serializable()


def test_empty_cloud_unverifiable():
    extents = np.array([0.30, 0.20, 0.25])
    res = verify_box_placement(np.zeros((0, 3)), _CENTER, _R, extents)
    assert res.verdict == UNVERIFIABLE
