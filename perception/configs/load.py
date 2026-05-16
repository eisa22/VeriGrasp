"""Load bottom-inference YAML configuration with code defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG: dict[str, Any] = {
    "lateral_radius_m": 0.30,
    "min_neighbors": 2,
    "height_tolerance": 0.005,
    "tolerance_m": 0.008,
    "pallet_height_tolerance": 0.030,
    "edge_distance_m": 0.05,
    "scene_pcd_stride": 2,
    "solid_surface": {
        "slab_m": 0.005,
        "min_points": 40,
        "min_fraction": 0.03,
        "min_step_m": 0.030,
        "neighbor_ring_outer_m": 0.10,
        "neighbor_ring_inner_m": 0.010,
        "min_component_pixels": 120,
        "min_aspect_ratio": 0.25,
        "raster_m": 0.005,
    },
    "obb": {
        "min_points": 50,
        "max_aspect_ratio": 20.0,
    },
    "audit": {
        "store_per_neighbor_distances": True,
        "store_neighbor_top_values": True,
    },
}

_CONFIG_PATH = Path(__file__).resolve().parent / "bottom_inference.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_bottom_inference_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config from YAML, merged over documented defaults."""
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    config_path = Path(path) if path is not None else _CONFIG_PATH
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, loaded)
    return cfg
