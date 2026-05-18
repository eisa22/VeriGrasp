"""Stage 2.5 orchestrator: bottom-plane inference for all candidates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from perception.bottom_inference.cases import decide_bottom
from perception.bottom_inference.neighbors import (
    build_geometry_index,
    find_lateral_neighbors,
    find_neighbor_via_gradient,
)
from perception.bottom_inference.obb import fit_extruded_obb
from perception.candidate import BottomInference, CandidateOut


def _resolve_neighbor_z(
    gradient_z: float | None,
    gradient_label: int | None,
    lateral_info,
) -> tuple[float | None, str | None, str]:
    """
    Pick the highest neighbour-surface from the gradient-based plateau
    analysis and the lateral-candidate fallback.
    """
    candidates_z: list[tuple[float, str | None, str]] = []
    if gradient_z is not None:
        gid = f"plateau_{gradient_label}" if gradient_label is not None else "plateau"
        candidates_z.append((gradient_z, gid, "gradient"))
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
    pallet_plane: tuple[float, float, float, float],
    config: dict,
    *,
    depth: np.ndarray | None = None,
    sobel_edges: np.ndarray | None = None,
    workspace_mask: np.ndarray | None = None,
    match_neighbors: list | None = None,
    scene_pcd: np.ndarray | None = None,
) -> list[CandidateOut]:
    """
    Enrich each candidate with bottom-plane inference.

    Algorithm per candidate (user spec):
      1. Lowest solid surface of the parcel from its SAM mask (z_visible_min).
      2. Around the parcel: define a neighbourhood (dilated ring) and use
         only Sobel/Canny gradient edges to split the ring into plateaus.
         The highest plateau whose top is below z_visible_min is the
         neighbour.
      3. If such a neighbour exists -> drop the bounding box to its top.
         Else, leave the box at z_visible_min (or fall to the pallet if
         the parcel sits directly on it).

    All heights are in 'above pallet' convention (z_0 = 0 = pallet).

    Args:
        depth: (H, W) absolute depth map. Required for gradient-based search.
        sobel_edges: (H, W) binary edge map (e.g. from Canny on depth).
        workspace_mask: (H, W) optional workspace gate.
        match_neighbors / scene_pcd: kept for API compatibility, not used by
            the gradient-only algorithm.
    """
    if not candidates:
        return candidates

    plane = tuple(float(x) for x in pallet_plane)
    geom_index = build_geometry_index(candidates, plane, config)

    enriched: list[CandidateOut] = []
    audit = config.get("audit", {})
    z_pallet = 0.0
    gradient_enabled = depth is not None and sobel_edges is not None

    for c in candidates:
        g = geom_index[c.candidate_id]

        if gradient_enabled:
            grad_result = find_neighbor_via_gradient(
                c, depth, sobel_edges, workspace_mask, plane,
                z_visible_min=g.z_visible_min, config=config,
            )
            gradient_z = grad_result.z_highest_neighbor
            gradient_label = grad_result.chosen_label
            gradient_plateaus = grad_result.plateaus
            gradient_ring_px = grad_result.n_ring_pixels
            if gradient_plateaus:
                plat_summary = ", ".join(
                    f"{p.height_above_pallet:+.3f}m({p.area_px}px)"
                    for p in sorted(
                        gradient_plateaus,
                        key=lambda x: -x.height_above_pallet,
                    )[:8]
                )
                print(
                    f"[BOTTOM-GR] {c.candidate_id[:8]} z_min={g.z_visible_min:.3f}m "
                    f"ring={gradient_ring_px}px plateaus={len(gradient_plateaus)} "
                    f"[{plat_summary}] -> picked={gradient_z if gradient_z is None else f'{gradient_z:.3f}m'}"
                )
        else:
            gradient_z = None
            gradient_label = None
            gradient_plateaus = []
            gradient_ring_px = 0

        lateral_info = find_lateral_neighbors(g, geom_index, config)

        z_highest_neighbor, highest_id, source = _resolve_neighbor_z(
            gradient_z, gradient_label, lateral_info,
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
        debug["n_lateral_neighbors"] = len(lateral_info.neighbor_ids)
        debug["neighbor_ids_used"] = list(decision.used_neighbor_ids)
        debug["z_visible_min"] = g.z_visible_min
        debug["z_neighbor_top_gradient"] = gradient_z
        debug["z_neighbor_top_lateral"] = lateral_info.z_neighbor_top
        debug["z_highest_neighbor"] = z_highest_neighbor
        debug["neighbor_source"] = source
        debug["highest_neighbor_id"] = highest_id
        debug["case_label"] = decision.case_label
        debug["center_xy"] = g.center_xy.tolist()
        debug["obb_extent_xy"] = g.obb_extent_xy
        debug["z_pallet"] = z_pallet
        debug["gradient_n_ring_pixels"] = gradient_ring_px
        debug["gradient_n_plateaus"] = len(gradient_plateaus)
        debug["gradient_plateaus"] = [
            {
                "label": p.label,
                "area_px": p.area_px,
                "height_above_pallet": p.height_above_pallet,
                "centroid_px": list(p.centroid_px),
            }
            for p in gradient_plateaus
        ]

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
