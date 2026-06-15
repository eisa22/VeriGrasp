"""Data types for the grasp verification module (Stage 13).

All margins are continuous floats (never plain booleans) so the records can be
used to build ROC curves (RQ2) and a combined soft score (RQ3). A positive
margin means the check passed with that much slack; a negative margin means it
failed by that amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckRecord:
    """Single verification check with its raw value, threshold and margin."""

    name: str
    stage: int
    raw_value: float
    threshold: float
    margin: float
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "raw_value": _safe_float(self.raw_value),
            "threshold": _safe_float(self.threshold),
            "margin": _safe_float(self.margin),
            "passed": bool(self.passed),
            "detail": _clean_detail(self.detail),
        }


@dataclass
class StageResult:
    """Outcome of one verification stage (a group of checks)."""

    stage: int
    name: str
    passed: bool
    checks: list[CheckRecord] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)

    def first_failed(self) -> CheckRecord | None:
        for c in self.checks:
            if not c.passed:
                return c
        return None

    def to_serializable(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "name": self.name,
            "passed": bool(self.passed),
            "checks": [c.to_serializable() for c in self.checks],
        }


@dataclass
class VerificationResult:
    """Aggregated verdict for a single grasp candidate."""

    verdict: str  # "ACCEPT" | "REJECT"
    mode: str  # "cascade" | "full"
    decisive_stage: int | None = None
    decisive_check: str | None = None
    stages: list[StageResult] = field(default_factory=list)
    soft_score: float | None = None
    candidate_id: str | None = None
    grasp_rank: int | None = None

    @property
    def accepted(self) -> bool:
        return self.verdict == "ACCEPT"

    def all_checks(self) -> list[CheckRecord]:
        out: list[CheckRecord] = []
        for s in self.stages:
            out.extend(s.checks)
        return out

    def to_serializable(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "mode": self.mode,
            "decisive_stage": self.decisive_stage,
            "decisive_check": self.decisive_check,
            "soft_score": _safe_float(self.soft_score) if self.soft_score is not None else None,
            "candidate_id": self.candidate_id,
            "grasp_rank": self.grasp_rank,
            "stages": [s.to_serializable() for s in self.stages],
        }

    def to_summary_serializable(self) -> dict[str, Any]:
        """Kompakte Verifikationsausgabe für Batch-Tests (ohne Stufen-Details)."""
        return {
            "verdict": self.verdict,
            "mode": self.mode,
            "decisive_stage": self.decisive_stage,
            "decisive_check": self.decisive_check,
            "soft_score": _safe_float(self.soft_score) if self.soft_score is not None else None,
            "candidate_id": self.candidate_id,
            "grasp_rank": self.grasp_rank,
        }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _clean_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy scalars/arrays in a detail dict to JSON-friendly values."""
    import numpy as np

    out: dict[str, Any] = {}
    for key, value in detail.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            out[key] = [
                v.item() if isinstance(v, np.generic) else v for v in value
            ]
        else:
            out[key] = value
    return out
