"""Experiment 6: threshold registry, raw-value recovery, and sweep grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

Direction = Literal["pass_if_leq", "pass_if_geq", "pass_if_lt"]
GridType = Literal["log", "linear", "integer"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = PROJECT_ROOT / "verification" / "verification.yaml"

# Hardware-fixed; not swept (Addendum).
EXCLUDED_CHECKS: tuple[dict[str, str], ...] = (
    {
        "check_param": "existence.min_points",
        "reason": "n_valid not in CSV; never binding at default (ratio-only gate)",
    },
    {
        "check_param": "corridor_clear.safety_corridor_height_m",
        "reason": "does not affect cascade pass (only n_blocking vs noise tolerance)",
    },
    {
        "check_param": "stage2.vacuum_pressure_pa",
        "reason": "hardware parameter; sweep safety factor only",
    },
)

CORRIDOR_INTEGER_GRID: tuple[int, ...] = (0, 1, 2, 3, 5, 7, 10, 15, 20)


@dataclass(frozen=True)
class ThresholdSpec:
    check_param: str
    check_name: str
    default: float
    unit: str
    direction: Direction
    domain_lo: float | None
    domain_hi: float | None
    grid_type: GridType
    raw_column: str | None = None  # e.g. n_blocking_points instead of margin


def _gripper_half_min(cfg: dict) -> float:
    g = cfg["gripper"]
    return min(float(g["width_m"]) * 0.5, float(g["length_m"]) * 0.5)


def _wrench_default_threshold(cfg: dict) -> float:
    s2 = cfg["stage2"]
    g = cfg["gripper"]
    vacuum_pa = float(s2["vacuum_pressure_pa"])
    pad_area = float(g["width_m"]) * float(g["length_m"])
    cup_reff = 0.5 * min(float(g["width_m"]), float(g["length_m"]))
    hold_safety = float(s2["holdability_safety_factor"])
    resist = vacuum_pa * pad_area * cup_reff
    return resist / hold_safety


def load_verification_thresholds(yaml_path: Path | None = None) -> dict:
    path = yaml_path or DEFAULT_YAML
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_threshold_specs(cfg: dict | None = None) -> list[ThresholdSpec]:
    cfg = cfg if cfg is not None else load_verification_thresholds()
    s1 = cfg["stage1"]
    bc = cfg["box_check"]
    s2 = cfg["stage2"]
    s3 = cfg["stage3"]
    edge_min = s2.get("edge_clearance_min_m")
    edge_default = float(edge_min) if edge_min is not None else _gripper_half_min(cfg)

    specs = [
        ThresholdSpec("existence.min_valid_fraction", "existence", float(s1["min_valid_fraction"]), "ratio", "pass_if_geq", 0.0, 1.0, "log"),
        ThresholdSpec("top_height_match.top_height_tol_m", "top_height_match", float(s1["top_height_tol_m"]), "m", "pass_if_leq", 0.0, None, "log"),
        ThresholdSpec("bbox_extent.extent_rel_dev_max", "bbox_extent", float(bc["extent_rel_dev_max"]), "rel_dev", "pass_if_leq", 0.0, None, "log"),
        ThresholdSpec("bbox_inlier.inlier_min", "bbox_inlier", float(bc["inlier_min"]), "ratio", "pass_if_geq", 0.0, 1.0, "log"),
        ThresholdSpec("bbox_surface_dist.surface_dist_median_max_m", "bbox_surface_dist", float(bc["surface_dist_median_max_m"]), "m", "pass_if_leq", 0.0, None, "log"),
        ThresholdSpec("bbox_top_normal.top_normal_angle_max_deg", "bbox_top_normal", float(bc["top_normal_angle_max_deg"]), "deg", "pass_if_leq", 0.0, 180.0, "log"),
        ThresholdSpec("bbox_coverage.min_coverage", "bbox_coverage", float(bc["min_coverage"]), "ratio", "pass_if_geq", 0.0, 1.0, "log"),
        ThresholdSpec("planarity.plane_rmse_max_m", "planarity", float(s2["plane_rmse_max_m"]), "m", "pass_if_leq", 0.0, None, "log"),
        ThresholdSpec("normal_angle.normal_angle_max_deg", "normal_angle", float(s2["normal_angle_max_deg"]), "deg", "pass_if_leq", 0.0, 180.0, "log"),
        ThresholdSpec("normal_scatter.normal_scatter_max", "normal_scatter", float(s2["normal_scatter_max"]), "ratio", "pass_if_leq", 0.0, None, "log"),
        ThresholdSpec("suction_area.min_area_ratio", "suction_area", float(s2["min_area_ratio"]), "ratio", "pass_if_geq", 0.0, 1.0, "linear"),
        ThresholdSpec("edge_clearance.edge_clearance_min_m", "edge_clearance", edge_default, "m", "pass_if_geq", 0.0, None, "log"),
        ThresholdSpec("data_gaps.max_empty_cell_fraction", "data_gaps", float(s2["max_empty_cell_fraction"]), "ratio", "pass_if_leq", 0.0, 1.0, "log"),
        ThresholdSpec("depth_seam.depth_seam_span_ratio", "depth_seam", float(s2["depth_seam_span_ratio"]), "ratio", "pass_if_lt", 0.0, 1.0, "log"),
        ThresholdSpec("surface_warp.max_peak_to_valley_m", "surface_warp", float(s2["max_peak_to_valley_m"]), "m", "pass_if_leq", 0.0, None, "log"),
        ThresholdSpec("normal_alignment.normal_alignment_max_deg", "normal_alignment", float(s2["normal_alignment_max_deg"]), "deg", "pass_if_leq", 0.0, 180.0, "log"),
        ThresholdSpec("surface_warp_robust.max_peak_to_valley_robust_m", "surface_warp_robust", float(s2["max_peak_to_valley_robust_m"]), "m", "pass_if_leq", 0.0, None, "log"),
        ThresholdSpec("suction_force.holdability_safety_factor", "suction_force", float(s2["holdability_safety_factor"]), "factor", "pass_if_geq", 0.0, None, "log"),
        ThresholdSpec("wrench_lever.moment_threshold", "wrench_lever", _wrench_default_threshold(cfg), "Nm", "pass_if_leq", 0.0, None, "log"),
        ThresholdSpec(
            "corridor_clear.noise_point_tolerance",
            "corridor_clear",
            float(s3["noise_point_tolerance"]),
            "count",
            "pass_if_leq",
            0.0,
            None,
            "integer",
            raw_column="n_blocking_points",
        ),
    ]
    return specs


def spec_by_param(specs: list[ThresholdSpec]) -> dict[str, ThresholdSpec]:
    return {s.check_param: s for s in specs}


def raw_value_from_margin(margin: np.ndarray, spec: ThresholdSpec) -> np.ndarray:
    m = np.asarray(margin, dtype=np.float64)
    d = spec.default
    if spec.direction in ("pass_if_leq", "pass_if_lt"):
        return d - m
    return m + d


def raw_values_for_spec(df: pd.DataFrame, spec: ThresholdSpec) -> np.ndarray:
    if spec.raw_column:
        if spec.raw_column not in df.columns:
            raise KeyError(f"missing column {spec.raw_column} — run patch_exp3_n_blocking.py first")
        return df[spec.raw_column].astype(float).values
    margin = df[f"check_{spec.check_name}_margin"].astype(float).values
    return raw_value_from_margin(margin, spec)


def pass_from_raw(raw: np.ndarray, tau: float, direction: Direction) -> np.ndarray:
    r = np.asarray(raw, dtype=np.float64)
    if direction == "pass_if_leq":
        return r <= tau
    if direction == "pass_if_lt":
        return r < tau
    return r >= tau


def pass_with_unverifiable(
    df: pd.DataFrame,
    spec: ThresholdSpec,
    tau: float,
) -> np.ndarray:
    raw = raw_values_for_spec(df, spec)
    verifiable = ~df[f"check_{spec.check_name}_unverifiable"].astype(bool).values
    computed = pass_from_raw(raw, tau, spec.direction)
    return computed | ~verifiable


def build_grid(spec: ThresholdSpec, *, n_points: int = 13) -> tuple[np.ndarray, bool, GridType]:
    default = spec.default
    if spec.grid_type == "integer":
        grid = np.array(CORRIDOR_INTEGER_GRID, dtype=float)
        return grid, False, "integer"

    if spec.grid_type == "linear":
        lo = 0.25 * default
        hi = 4.0 * default
        if spec.domain_lo is not None:
            lo = max(lo, spec.domain_lo)
        if spec.domain_hi is not None:
            hi = min(hi, spec.domain_hi)
        grid = np.linspace(lo, hi, n_points)
    else:
        lo = max(0.25 * default, spec.domain_lo if spec.domain_lo is not None else 0.0)
        hi = 4.0 * default
        if spec.domain_hi is not None:
            hi = min(hi, spec.domain_hi)
        if lo <= 0:
            lo = default * 0.25 if default > 0 else 1e-6
        grid = np.geomspace(max(lo, 1e-12), max(hi, lo * 1.01), n_points)

    if default not in grid:
        grid = np.sort(np.unique(np.append(grid, default)))
    grid = np.sort(np.unique(grid))

    clamped = False
    if spec.domain_lo is not None:
        before = grid.copy()
        grid = grid[grid >= spec.domain_lo - 1e-15]
        if len(grid) < len(before):
            clamped = True
    if spec.domain_hi is not None:
        before = grid.copy()
        grid = grid[grid <= spec.domain_hi + 1e-15]
        if len(grid) < len(before):
            clamped = True

    return grid, clamped, spec.grid_type


def _clamp_grid_to_domain(grid: np.ndarray, spec: ThresholdSpec) -> np.ndarray:
    out = np.sort(np.unique(np.asarray(grid, dtype=np.float64)))
    if spec.domain_lo is not None:
        out = out[out >= spec.domain_lo - 1e-15]
    if spec.domain_hi is not None:
        out = out[out <= spec.domain_hi + 1e-15]
    return np.sort(np.unique(out))


def build_pairwise_grid(spec: ThresholdSpec, n: int = 7) -> np.ndarray:
    lo, hi = 0.5 * spec.default, 2.0 * spec.default
    if spec.grid_type == "integer":
        ints = sorted(set(int(round(x)) for x in np.linspace(lo, hi, n)))
        if int(spec.default) not in ints:
            ints.append(int(spec.default))
        grid = np.array(sorted(ints), dtype=float)
    elif spec.grid_type == "linear":
        grid = np.linspace(lo, hi, n)
    else:
        grid = np.geomspace(max(lo, 1e-12), max(hi, lo * 1.01), n)
    if spec.default not in grid:
        grid = np.sort(np.unique(np.append(grid, spec.default)))
    return _clamp_grid_to_domain(grid, spec)


def mapping_table_json(specs: list[ThresholdSpec]) -> list[dict]:
    out: list[dict] = []
    for s in specs:
        out.append({
            "check_param": s.check_param,
            "unit": s.unit,
            "direction": s.direction,
            "default": s.default,
            "domain": [s.domain_lo, s.domain_hi],
            "grid_type": s.grid_type,
        })
    return out


def validate_round_trip(df: pd.DataFrame, specs: list[ThresholdSpec]) -> dict[str, int]:
    """Per sweep parameter: mismatches on verifiable rows at default threshold."""
    mismatches: dict[str, int] = {}
    for spec in specs:
        stored = df[f"check_{spec.check_name}_pass"].astype(bool).values
        recomputed = pass_with_unverifiable(df, spec, spec.default)
        verifiable = ~df[f"check_{spec.check_name}_unverifiable"].astype(bool).values
        mm = int(((recomputed != stored) & verifiable).sum())
        mismatches[spec.check_param] = mm
    return mismatches
