"""Candidate output schema for the perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BottomInference:
    bottom_z: float
    bottom_method: str
    bottom_confidence: float
    bottom_residual_m: float
    used_neighbor_ids: list[str]
    height_m: float
    parcel_obb: dict


@dataclass
class CandidateOut:
    candidate_id: str
    mask_2d: np.ndarray
    points_3d: np.ndarray
    centroid_3d: np.ndarray
    surface_normal: np.ndarray
    surface_area_m2: float
    top_surface_height: float
    bbox_2d: tuple[int, int, int, int]
    debug: dict = field(default_factory=dict)
    bottom: BottomInference | None = None
