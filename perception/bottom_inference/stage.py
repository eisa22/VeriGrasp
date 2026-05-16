"""Stage 2.5 orchestrator: bottom-plane inference for all candidates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from perception.bottom_inference.cases import decide_bottom
from perception.bottom_inference.neighbors import (
    build_geometry_index,
    find_lateral_neighbors,
    find_neighbor_surface_via_scene,
)
from perception.bottom_inference.obb import fit_extruded_obb
from perception.candidate import BottomInference, CandidateOut


def _resolve_neighbor_z(
    target_geom,
    scene_z: float | None,
    lateral_info,
) -> tuple[float | None, str | None, str]:
    """
    Combine scene-based neighbour search and lateral candidate search.

    Priority: take the HIGHEST surface among both sources (closest to our
    parcel's bottom but still below it).
    """
    candidates_z: list[tuple[float, str | None, str]] = []
    if scene_z is not None:
        candidates_z.append((scene_z, "scene", "scene"))
    if lateral_info.z_highest_neighbor is not None:
        candidates_z.append((
            float(lateral_info.z_highest_neighbor),
            lateral_info.highest_neighbor_id,
            "lateral",
        ))
    if not candidates_z:
        return None, None, ""
    best = max(candidates_z, key=lambda x: x[0])
    return best[0], best[1], best[2]


def infer_bottom_planes(
    candidates: list[CandidateOut],
    scene_pcd: np.ndarray,
    pallet_plane: tuple[float, float, float, float],
    config: dict,
) -> list[CandidateOut]:
    """
    Enrich each candidate with bottom-plane inference.

    Algorithm per candidate:
      1. Find lowest solid surface of the parcel (z_visible_min).
      2. Search scene point cloud inside the OBB footprint for the highest
         solid surface strictly below the parcel (the actual neighbour).
      3. Also consider detected candidates as lateral neighbours.
      4. If neighbour < z_visible_min - tolerance -> extend box down.
         Else -> leave at z_visible_min.

    All heights are in 'above pallet' convention (z_0 = 0 = pallet).
    """
    if not candidates:
        return candidates

    plane = tuple(float(x) for x in pallet_plane)
    geom_index = build_geometry_index(candidates, plane, config)

    enriched: list[CandidateOut] = []
    audit = config.get("audit", {})
    z_pallet = 0.0

    for c in candidates:
        g = geom_index[c.candidate_id]

        scene_z, scene_n = find_neighbor_surface_via_scene(g, scene_pcd, plane, config)
        lateral_info = find_lateral_neighbors(g, geom_index, config)

        z_highest_neighbor, highest_id, source = _resolve_neighbor_z(
            g, scene_z, lateral_info,
        )

        decision = decide_bottom(
            target=g,
            z_highest_neighbor=z_highest_neighbor,
            highest_neighbor_id=highest_id,
            neighbor_source=source,
            z_pallet=z_pallet,
            config=config,
        )

        top_z = float(c.top_surface_height)
        bottom_z = float(decision.bottom_z)
        residual = abs(g.z_visible_min - bottom_z)
        height_m = max(top_z - bottom_z, 0.0)

        parcel_obb = fit_extruded_obb(c, plane, bottom_z, top_z, config)

        bottom = BottomInference(
            bottom_z=bottom_z,
            bottom_method=decision.bottom_method,
            bottom_confidence=decision.bottom_confidence,
            bottom_residual_m=residual,
            used_neighbor_ids=list(decision.used_neighbor_ids),
            height_m=height_m,
            parcel_obb=parcel_obb,
        )

        debug = dict(c.debug)
        debug["n_neighbors_found"] = len(lateral_info.neighbor_ids)
        debug["n_scene_points_below"] = scene_n
        debug["neighbor_ids_used"] = list(decision.used_neighbor_ids)
        debug["neighbor_spread_m"] = lateral_info.neighbor_spread
        debug["z_visible_min"] = g.z_visible_min
        debug["z_neighbor_top_lateral"] = lateral_info.z_neighbor_top
        debug["z_neighbor_top_scene"] = scene_z
        debug["z_highest_neighbor"] = z_highest_neighbor
        debug["neighbor_source"] = source
        debug["highest_neighbor_id"] = highest_id
        debug["case_label"] = decision.case_label
        debug["center_xy"] = g.center_xy.tolist()
        debug["obb_extent_xy"] = g.obb_extent_xy
        debug["z_pallet"] = z_pallet

        if audit.get("store_per_neighbor_distances", False):
            debug["neighbor_distances_m"] = list(lateral_info.neighbor_distances)
        if audit.get("store_neighbor_top_values", False):
            debug["neighbor_top_values_m"] = list(lateral_info.neighbor_tops)

        enriched.append(
            replace(
                c,
                bottom=bottom,
                debug=debug,
            )
        )

    if config.get("debug_viz"):
        from perception.bottom_inference.debug import visualize_bottom_inference
        visualize_bottom_inference(enriched, plane)

    return enriched
