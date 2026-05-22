"""Load bottom-inference YAML configuration with code defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG: dict[str, Any] = {
    "lateral_radius_m": 0.30,
    "lateral_radius_factor": 1.5,
    "lateral_radius_max_m": 1.5,
    "min_neighbors": 2,
    "height_tolerance": 0.005,
    "tolerance_m": 0.008,
    "pallet_height_tolerance": 0.030,
    "edge_distance_m": 0.05,
    "global_gradient_plateaus": {
        "enabled": True,
        "split_height_bands": True,
        "edge_dilate_px": 1,
        "exclude_dilate_px": 4,
        "min_component_px": 60,
        "min_band_pixels": 30,
        "min_band_fraction": 0.01,
        "min_plateau_area_px": 40,
        "min_plateau_area_m2": 0.0005,
        "max_plateau_z_std_m": 0.030,
        "min_aspect_ratio": 0.03,
        "search_pad_m": 0.35,
        "min_overlap": 0.01,
        "max_centroid_dist_m": 0.55,
    },
    "gradient_neighbor": {
        "neighbor_radius_m": 0.30,
        "adaptive_radius_factor": 1.0,
        "max_radius_m": 1.0,
        "neighbor_radius_px": 80,
        "min_plateau_area_px": 300,
        "min_plateau_area_m2": 0.005,
        "max_plateau_z_std_m": 0.015,
        "min_aspect_ratio": 0.25,
        "use_dominant_height_band": True,
        "slab_m": 0.005,
        "edge_dilate_px": 1,
        "height_percentile": 95.0,
    },
    "depth_histogram": {
        "enabled": True,
        "slab_m": 0.005,
        "min_band_pixels": 200,
        "min_band_fraction": 0.02,
        "min_component_area_m2": 0.003,
        "min_component_aspect": 0.20,
    },
    "scene_planes": {
        "enabled": True,
        "edge_dilate_px": 2,
        "exclude_dilate_px": 4,
        "min_component_px": 200,
        "min_area_m2": 0.003,
        "min_aspect_ratio": 0.15,
        "max_height_std_m": 0.020,
        "slab_m": 0.005,
        "min_band_pixels": 100,
        "min_band_fraction": 0.05,
        "min_overlap": 0.05,
        "max_centroid_dist_m": 0.20,
    },
    "solid_surface": {
        "slab_m": 0.005,
        "min_points": 40,
        "min_fraction": 0.03,
        "raster_m": 0.005,
        "min_component_pixels": 120,
        "min_aspect_ratio": 0.25,
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


_DEFAULT_SUCTION_GRASP_CONFIG: dict[str, Any] = {
    "backend": "normal_std",
    "max_grasps": 20,
    "min_score": 0.3,
    "grid_down_rate": 10,
    "grid_topk": 1024,
    "heatmap_kernel_size": 15,
    "min_separation_m": 0.04,
    "normal_knn": 224,
    "normal_std_filter_size": 25,
    "fx": 437.04,
    "fy": 437.04,
    "model": "deeplabv3plus_resnet101",
    "checkpoint_path": None,
    "depth_clamp_max_m": 3.0,
}

_SUCTION_GRASP_CONFIG_PATH = Path(__file__).resolve().parent / "suction_grasp.yaml"


def load_suction_grasp_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load suction grasp config from YAML, merged over documented defaults."""
    cfg = copy.deepcopy(_DEFAULT_SUCTION_GRASP_CONFIG)
    config_path = Path(path) if path is not None else _SUCTION_GRASP_CONFIG_PATH
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, loaded)
    return cfg
