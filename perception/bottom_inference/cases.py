"""
Bottom-plane decision logic.

Algorithm (heights in 'above pallet' convention, z_0 = 0 = pallet):

    1. z_lowest_visible = lowest SOLID surface of the parcel.
    2. z_highest_neighbor = highest SOLID surface strictly below the
       parcel (from the scene point cloud or detected candidates).
    3. If z_highest_neighbor < z_lowest_visible - tolerance:
           neighbour is closer to pallet than what we can see.
           -> Extend the bounding box DOWN to the neighbour plane.
       Else:
           Leave the bounding box at the lowest visible surface.

Fallbacks (no neighbour at all):
    - close to pallet -> from_pallet (drop box down to pallet)
    - otherwise        -> uncertain (leave at lowest visible)
"""

from __future__ import annotations

from dataclasses import dataclass

from perception.bottom_inference.neighbors import CandidateGeometry


@dataclass
class BottomDecision:
    bottom_z: float
    bottom_method: str
    bottom_confidence: float
    case_label: str
    used_neighbor_ids: list[str]


def _clip_confidence(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def decide_bottom(
    target: CandidateGeometry,
    z_highest_neighbor: float | None,
    highest_neighbor_id: str | None,
    neighbor_source: str,
    z_pallet: float,
    config: dict,
) -> BottomDecision:
    delta = float(config["tolerance_m"])
    pallet_tol = float(config["pallet_height_tolerance"])

    z_lowest_visible = float(target.z_visible_min)

    if z_highest_neighbor is None:
        if z_lowest_visible - z_pallet <= pallet_tol:
            return BottomDecision(
                bottom_z=z_pallet,
                bottom_method="from_pallet",
                bottom_confidence=0.7,
                case_label="Fallback_pallet",
                used_neighbor_ids=[],
            )
        return BottomDecision(
            bottom_z=z_lowest_visible,
            bottom_method="uncertain",
            bottom_confidence=0.3,
            case_label="Fallback_none",
            used_neighbor_ids=[],
        )

    used_ids = [highest_neighbor_id] if highest_neighbor_id is not None else []

    if z_highest_neighbor < z_lowest_visible - delta:
        confidence = 0.90 if neighbor_source == "gradient" else 0.75
        return BottomDecision(
            bottom_z=z_highest_neighbor,
            bottom_method="from_neighbor",
            bottom_confidence=confidence,
            case_label=f"B_dropped_to_{neighbor_source or 'unknown'}",
            used_neighbor_ids=used_ids,
        )

    confidence = 0.9 if abs(z_highest_neighbor - z_lowest_visible) <= delta else 0.6
    return BottomDecision(
        bottom_z=z_lowest_visible,
        bottom_method="measured",
        bottom_confidence=confidence,
        case_label="A_measured",
        used_neighbor_ids=used_ids,
    )
