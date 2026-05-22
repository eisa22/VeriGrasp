"""Data types for Stage 11 suction grasp generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SuctionGrasp:
    """Single vacuum suction grasp candidate in camera coordinates."""

    score: float
    normal: np.ndarray
    position: np.ndarray
    row: int
    col: int
    rank: int = 0

    def to_serializable(self) -> dict:
        return {
            "score": float(self.score),
            "normal": list(map(float, self.normal.tolist())),
            "position": list(map(float, self.position.tolist())),
            "pixel": [int(self.row), int(self.col)],
            "rank": int(self.rank),
        }


@dataclass
class SuctionGraspResult:
    """Outcome of Stage 11 for the selected target."""

    grasps: list[SuctionGrasp]
    candidate_id: str | None
    backend: str
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "backend": self.backend,
            "n_grasps": len(self.grasps),
            "grasps": [g.to_serializable() for g in self.grasps],
            "config": self.config_snapshot,
            "debug": self.debug,
        }
