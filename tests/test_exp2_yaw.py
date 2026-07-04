"""Yaw folding unit tests (Experiment 2 acceptance criteria, spec 9.2)."""

import numpy as np
import pytest

from evaluation.exp2_geometry import angle_between_deg, orient_toward_camera, yaw_error_deg


def test_yaw_fold_180_wraparound():
    # psi_pred = 179 deg, psi_gt = 1 deg -> e_yaw = 2 deg (spec unit test)
    err, fold = yaw_error_deg(179.0, 1.0, gt_aspect=2.0)
    assert fold == 180
    assert err == pytest.approx(2.0)


def test_yaw_fold_90_near_square():
    # near-square box: psi_pred = 46, psi_gt = -44 -> e_yaw = 0 (spec unit test)
    err, fold = yaw_error_deg(46.0, -44.0, gt_aspect=1.0)
    assert fold == 90
    assert err == pytest.approx(0.0)


def test_yaw_fold_180_same_angle():
    err, fold = yaw_error_deg(30.0, 30.0, gt_aspect=3.0)
    assert fold == 180
    assert err == pytest.approx(0.0)


def test_yaw_fold_max_error():
    # 90 deg apart under 180-fold is the maximum possible error.
    err, _ = yaw_error_deg(90.0, 0.0, gt_aspect=2.0)
    assert err == pytest.approx(90.0)


def test_yaw_fold_threshold_boundary():
    # aspect exactly at 1.2 uses the 180-deg fold (strict '<' for near-square)
    _, fold = yaw_error_deg(0.0, 0.0, gt_aspect=1.2)
    assert fold == 180
    _, fold = yaw_error_deg(0.0, 0.0, gt_aspect=1.19)
    assert fold == 90


def test_angle_between_identical_is_zero():
    n = np.array([0.1, -0.2, -0.97])
    assert angle_between_deg(n, n) == pytest.approx(0.0, abs=1e-9)


def test_angle_between_orthogonal():
    assert angle_between_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)


def test_orient_toward_camera_flips_positive_z():
    n = orient_toward_camera(np.array([0.0, 0.0, 1.0]))
    assert n[2] == pytest.approx(-1.0)
    n = orient_toward_camera(np.array([0.0, 0.0, -1.0]))
    assert n[2] == pytest.approx(-1.0)
