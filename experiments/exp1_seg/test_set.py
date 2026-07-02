"""Named test subsets for Experiment 1 (--test-set)."""

from __future__ import annotations

from pathlib import Path

from evaluation.scene_registry import scene_index_from_id


def list_test_set_names(eval_cfg: dict) -> list[str]:
    sets = eval_cfg.get("test_sets") or {}
    return sorted(sets.keys())


def resolve_scene_ids(test_set_name: str, eval_cfg: dict) -> list[str]:
    sets = eval_cfg.get("test_sets") or {}
    if test_set_name not in sets:
        available = ", ".join(list_test_set_names(eval_cfg)) or "(none)"
        raise ValueError(f"Unknown test set '{test_set_name}'. Available: {available}")

    spec = sets[test_set_name]
    if "scene_ids" in spec:
        return [str(s) for s in spec["scene_ids"]]

    if "scene_indices" in spec:
        return [f"scene_{int(i):03d}" for i in spec["scene_indices"]]

    if "index_range" in spec:
        lo, hi = spec["index_range"]
        return [f"scene_{i:03d}" for i in range(int(lo), int(hi) + 1)]

    raise ValueError(f"Test set '{test_set_name}' has no scene_ids, scene_indices, or index_range")


def filter_session_paths(session_paths: list[str], test_set_name: str, eval_cfg: dict) -> list[str]:
    wanted = set(resolve_scene_ids(test_set_name, eval_cfg))
    out = [p for p in session_paths if Path(p).name in wanted]
    missing = wanted - {Path(p).name for p in out}
    if missing:
        print(f"[TEST-SET] Warning: {len(missing)} scene(s) not found in dataset: {sorted(missing)[:5]}...")
    return out


def filter_scene_id_files(paths: list[Path], test_set_name: str, eval_cfg: dict) -> list[Path]:
    wanted = set(resolve_scene_ids(test_set_name, eval_cfg))
    return [p for p in paths if p.stem in wanted]


def test_set_manifest(test_set_name: str, eval_cfg: dict, scene_ids: list[str]) -> dict:
    sets = eval_cfg.get("test_sets") or {}
    spec = sets.get(test_set_name, {})
    return {
        "name": test_set_name,
        "description": spec.get("description", ""),
        "n_scenes": len(scene_ids),
        "scene_ids": scene_ids,
    }
