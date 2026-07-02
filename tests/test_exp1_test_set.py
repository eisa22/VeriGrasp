"""Tests for Experiment 1 test-set resolution."""

from experiments.exp1_seg.test_set import (
    list_test_set_names,
    resolve_scene_ids,
)


def _cfg():
    return {
        "test_sets": {
            "smoke": {"scene_indices": [0, 1, 2]},
            "baseline": {"index_range": [0, 2]},
            "diverse": {"scene_ids": ["scene_000", "scene_100"]},
        }
    }


def test_list_names():
    assert "smoke" in list_test_set_names(_cfg())


def test_resolve_indices():
    ids = resolve_scene_ids("smoke", _cfg())
    assert ids == ["scene_000", "scene_001", "scene_002"]


def test_resolve_range():
    ids = resolve_scene_ids("baseline", _cfg())
    assert ids == ["scene_000", "scene_001", "scene_002"]


def test_resolve_explicit_ids():
    ids = resolve_scene_ids("diverse", _cfg())
    assert ids == ["scene_000", "scene_100"]
