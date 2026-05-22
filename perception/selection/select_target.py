"""Pick the next-action target from enriched candidates (Stage 10).

Priority (strongest → weakest):
  1. Top layer only: `top_surface_height` within `top_band_m` of scene max (5 cm).
  2. Smallest 3D-OBB Z-extent (`height_m`) within that tier.
  3. Fewest lateral parcel neighbours (exposed / outer-edge preference).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perception.candidate import CandidateOut

DEFAULT_TOP_BAND_M = 0.05
DEFAULT_NEIGHBOR_RADIUS_M = 0.30
SELECTION_POLICY = "highest_tier_smallest_extent_fewest_neighbors"


@dataclass
class SelectedTarget:
    """A single candidate that has been selected as the next action target."""
    candidate: CandidateOut
    rank: int
    score: float
    score_name: str = "z_extent_m"
    reason: str = SELECTION_POLICY
    n_lateral_peers: int = 0


@dataclass
class SelectionResult:
    """Outcome of the selection stage."""
    primary: SelectedTarget | None
    ranking: list[SelectedTarget] = field(default_factory=list)
    rejected: list[tuple[CandidateOut, str]] = field(default_factory=list)
    max_top_m: float | None = None
    top_band_m: float = DEFAULT_TOP_BAND_M
    neighbor_radius_m: float = DEFAULT_NEIGHBOR_RADIUS_M

    @property
    def primary_id(self) -> str | None:
        return self.primary.candidate.candidate_id if self.primary else None

    def to_serializable(self) -> dict:
        """JSON-friendly summary for downstream stages."""
        def _candidate_summary(c: CandidateOut) -> dict:
            b = c.bottom
            return {
                "candidate_id": c.candidate_id,
                "label": c.debug.get("label", c.candidate_id[:6]),
                "top_z": float(c.top_surface_height),
                "bottom_z": float(b.bottom_z) if b else None,
                "height_m": float(b.height_m) if b else None,
                "bottom_method": b.bottom_method if b else None,
                "bottom_confidence": float(b.bottom_confidence) if b else None,
                "centroid_3d": list(map(float, c.centroid_3d.tolist())),
                "parcel_obb": b.parcel_obb if b else None,
            }

        return {
            "selection_policy": SELECTION_POLICY,
            "max_top_m": self.max_top_m,
            "top_band_m": self.top_band_m,
            "neighbor_radius_m": self.neighbor_radius_m,
            "primary": {
                "candidate": _candidate_summary(self.primary.candidate),
                "rank": self.primary.rank,
                "score": self.primary.score,
                "score_name": self.primary.score_name,
                "reason": self.primary.reason,
                "n_lateral_peers": self.primary.n_lateral_peers,
            } if self.primary else None,
            "ranking": [
                {
                    "candidate_id": t.candidate.candidate_id,
                    "rank": t.rank,
                    "score": t.score,
                    "top_z": float(t.candidate.top_surface_height),
                    "n_lateral_peers": t.n_lateral_peers,
                }
                for t in self.ranking
            ],
            "rejected": [
                {"candidate_id": c.candidate_id, "reason": reason}
                for c, reason in self.rejected
            ],
        }


def _center_xy(c: CandidateOut) -> np.ndarray:
    if "center_xy" in c.debug:
        return np.asarray(c.debug["center_xy"], dtype=np.float64)
    if len(c.points_3d) > 0:
        return np.asarray(c.points_3d, dtype=np.float64)[:, :2].mean(axis=0)
    return np.zeros(2, dtype=np.float64)


def _obb_extent_xy(c: CandidateOut) -> float:
    if "obb_extent_xy" in c.debug:
        return float(c.debug["obb_extent_xy"])
    xy = np.asarray(c.points_3d, dtype=np.float64)[:, :2] if len(c.points_3d) else None
    if xy is None or len(xy) == 0:
        return 0.15
    return float(max(xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min()))


def count_lateral_peers(
    target: CandidateOut,
    pool: list[CandidateOut],
    *,
    neighbor_radius_m: float = DEFAULT_NEIGHBOR_RADIUS_M,
) -> int:
    """Count other parcels whose XY centre lies within the lateral search disc.

    Same radius model as bottom-inference lateral neighbours:
    r = neighbor_radius_m + 0.5 * target OBB extent.
    Fewer peers ⇒ parcel sits more on the pallet outer edge.
    """
    center = _center_xy(target)
    r = neighbor_radius_m + 0.5 * _obb_extent_xy(target)
    n = 0
    for other in pool:
        if other.candidate_id == target.candidate_id:
            continue
        if float(np.linalg.norm(center - _center_xy(other))) <= r:
            n += 1
    return n


def _filter_eligible(
    candidates: list[CandidateOut],
    *,
    min_confidence: float,
    require_method: tuple[str, ...] | None,
) -> tuple[list[CandidateOut], list[tuple[CandidateOut, str]]]:
    eligible: list[CandidateOut] = []
    rejected: list[tuple[CandidateOut, str]] = []

    for c in candidates:
        if c.bottom is None:
            rejected.append((c, "no_bottom_inference"))
            continue
        if c.bottom.bottom_confidence < min_confidence:
            rejected.append((
                c,
                f"low_confidence ({c.bottom.bottom_confidence:.2f} < {min_confidence:.2f})",
            ))
            continue
        if require_method and c.bottom.bottom_method not in require_method:
            rejected.append((
                c,
                f"method '{c.bottom.bottom_method}' not in {list(require_method)}",
            ))
            continue
        eligible.append(c)

    return eligible, rejected


def select_target_smallest_z(
    candidates: list[CandidateOut],
    *,
    min_confidence: float = 0.5,
    require_method: tuple[str, ...] | None = None,
    top_band_m: float = DEFAULT_TOP_BAND_M,
    neighbor_radius_m: float = DEFAULT_NEIGHBOR_RADIUS_M,
) -> SelectionResult:
    """Select target using three-tier priority (see module docstring)."""
    eligible, rejected = _filter_eligible(
        candidates,
        min_confidence=min_confidence,
        require_method=require_method,
    )

    if not eligible:
        return SelectionResult(
            primary=None,
            ranking=[],
            rejected=rejected,
            top_band_m=top_band_m,
            neighbor_radius_m=neighbor_radius_m,
        )

    max_top = max(float(c.top_surface_height) for c in eligible)
    cutoff = max_top - top_band_m

    top_tier: list[CandidateOut] = []
    for c in eligible:
        top = float(c.top_surface_height)
        if top >= cutoff - 1e-9:
            top_tier.append(c)
        else:
            rejected.append((
                c,
                f"below_top_tier (top={top:.3f}m < cutoff={cutoff:.3f}m, "
                f"max_top={max_top:.3f}m, band={top_band_m:.3f}m)",
            ))

    if not top_tier:
        return SelectionResult(
            primary=None,
            ranking=[],
            rejected=rejected,
            max_top_m=max_top,
            top_band_m=top_band_m,
            neighbor_radius_m=neighbor_radius_m,
        )

    peer_counts = {
        c.candidate_id: count_lateral_peers(c, top_tier, neighbor_radius_m=neighbor_radius_m)
        for c in top_tier
    }

    top_tier.sort(key=lambda c: (
        float(c.bottom.height_m),
        peer_counts[c.candidate_id],
        c.candidate_id,
    ))

    ranking = [
        SelectedTarget(
            candidate=c,
            rank=i,
            score=float(c.bottom.height_m),
            score_name="z_extent_m",
            reason=SELECTION_POLICY,
            n_lateral_peers=peer_counts[c.candidate_id],
        )
        for i, c in enumerate(top_tier)
    ]

    return SelectionResult(
        primary=ranking[0],
        ranking=ranking,
        rejected=rejected,
        max_top_m=max_top,
        top_band_m=top_band_m,
        neighbor_radius_m=neighbor_radius_m,
    )
