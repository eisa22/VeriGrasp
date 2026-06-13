"""Grasp verification module (Stage 13).

Deterministic three-stage gate on the primary suction grasp: bounding-box
validity, suctionability of the grasp point, and a clear vertical lift corridor.
Every check yields a continuous margin and an audit record.
"""

from verification.config import load_verification_config, resolve_gripper, GripperFootprint
from verification.verify import verify_grasp

__all__ = [
    "verify_grasp",
    "load_verification_config",
    "resolve_gripper",
    "GripperFootprint",
    "VerificationResult",
    "StageResult",
    "CheckRecord",
]
