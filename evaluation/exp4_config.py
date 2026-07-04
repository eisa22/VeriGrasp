"""Experiment 4: criterion mapping and shared check order."""

from __future__ import annotations

from evaluation.exp3_offline_verify import CHECK_ORDER

CRITERION_MAPPING_VERSION = "thesis_tab_verif_mapping+v1_force_estimate"

CRITERION_GROUPS: dict[str, tuple[str, ...]] = {
    "surface_planarity": ("planarity", "surface_warp", "surface_warp_robust"),
    "normal_consistency": ("normal_angle", "normal_scatter", "normal_alignment"),
    "depth_quality": ("existence", "data_gaps", "depth_seam"),
    "min_suction_area": ("suction_area", "edge_clearance"),
    "corridor_clearance": ("corridor_clear",),
    "segmentation_consistency": (
        "top_height_match",
        "bbox_extent",
        "bbox_inlier",
        "bbox_surface_dist",
        "bbox_top_normal",
        "bbox_coverage",
    ),
    # Not in thesis Table tab:verif:mapping; keeps all 20 checks covered.
    "force_estimate": ("suction_force", "wrench_lever"),
}

# Checks targeting sensor artefacts that synthetic clean depth cannot produce.
SENSOR_ARTIFACT_CHECKS: frozenset[str] = frozenset({"data_gaps", "depth_seam"})

assert set(CHECK_ORDER) == {c for grp in CRITERION_GROUPS.values() for c in grp}
