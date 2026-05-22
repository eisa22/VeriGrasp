"""Stage 2.5 orchestrator: bottom-plane inference for all candidates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from perception.bottom_inference.cases import decide_bottom
from perception.bottom_inference.neighbors import (
    build_geometry_index,
    detect_global_gradient_plateaus,
    find_lateral_neighbors,
    find_neighbor_from_gradient_catalog,
    find_neighbor_via_depth_histogram,
    find_neighbor_via_gradient,
)
from perception.bottom_inference.obb import fit_extruded_obb
from perception.bottom_inference.scene_planes import (
    detect_scene_planes,
    find_neighbor_via_scene_planes,
)
from perception.candidate import BottomInference, CandidateOut


def _resolve_neighbor_z(
    gradient_z: float | None,
    gradient_label: int | None,
    gradient_global_z: float | None,
    gradient_global_id: int | None,
    histogram_z: float | None,
    scene_z: float | None,
    scene_id: int | None,
    lateral_info,
) -> tuple[float | None, str | None, str]:
    """Pick the highest valid neighbour across all sources."""
    candidates_z: list[tuple[float, str | None, str]] = []
    if gradient_global_z is not None:
        gid = (
            f"global_plateau_{gradient_global_id}"
            if gradient_global_id is not None
            else "global_plateau"
        )
        candidates_z.append((gradient_global_z, gid, "gradient_global"))
    if gradient_z is not None:
        gid = f"plateau_{gradient_label}" if gradient_label is not None else "plateau"
        candidates_z.append((gradient_z, gid, "gradient"))
    if histogram_z is not None:
        candidates_z.append((histogram_z, "depth_band", "histogram"))
    if scene_z is not None:
        sid = f"scene_plane_{scene_id}" if scene_id is not None else "scene_plane"
        candidates_z.append((scene_z, sid, "scene_plane"))
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
    return_scene_planes: bool = False,
    return_gradient_catalog: bool = False,
) -> list[CandidateOut]:
    """
    Enrich each candidate with bottom-plane inference.

    Algorithm per candidate (user spec):
      1. Lowest solid surface of the parcel from its SAM mask (z_visible_min).
      2. Two parallel neighbour sources:
         (a) Gradient: ring around the parcel split into plateaus by
             Sobel/Canny edges. Highest plateau with top < z_visible_min.
         (b) Lateral: top-surfaces of OTHER detected candidates within
             `lateral_radius_m`. Highest top below z_visible_min.
         The HIGHER of the two qualifying surfaces wins.
      3. If a neighbour exists -> drop the bounding box to its top.
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
    histogram_enabled = depth is not None
    scene_enabled = depth is not None

    if scene_enabled:
        exclude_masks = [c.mask_2d for c in candidates]
        scene_planes = detect_scene_planes(
            depth=depth,
            sobel_edges=sobel_edges,
            workspace_mask=workspace_mask,
            exclude_masks=exclude_masks,
            plane=plane,
            config=config,
        )
        print(
            f"[BOTTOM-SP] detected {len(scene_planes)} scene planes "
            f"(non-candidate flat regions)"
        )
        for sp in sorted(scene_planes, key=lambda p: -p.height_above_pallet)[:12]:
            print(
                f"             id={sp.plane_id} h={sp.height_above_pallet:+.3f}m "
                f"A={sp.area_m2*1e4:.0f}cm² σ={sp.height_std_m*1000:.0f}mm "
                f"ar={sp.aspect_ratio:.2f}"
            )
    else:
        scene_planes = []

    global_gradient_catalog: list = []
    if gradient_enabled:
        exclude_masks = [c.mask_2d for c in candidates]
        global_gradient_catalog = detect_global_gradient_plateaus(
            depth=depth,
            sobel_edges=sobel_edges,
            workspace_mask=workspace_mask,
            exclude_masks=exclude_masks,
            plane=plane,
            config=config,
        )
        print(
            f"[BOTTOM-GG] global gradient catalogue: "
            f"{len(global_gradient_catalog)} plateaus (full workspace, Sobel-split)"
        )
        for gp in sorted(
            global_gradient_catalog, key=lambda p: -p.height_above_pallet,
        )[:12]:
            print(
                f"             id={gp.global_id} h={gp.height_above_pallet:+.3f}m "
                f"A={gp.area_m2*1e4:.0f}cm² σ={gp.height_std_m*1000:.0f}mm "
                f"ar={gp.aspect_ratio:.2f}"
            )

    for c in candidates:
        g = geom_index[c.candidate_id]
        gradient_global_z = None
        gradient_global_id = None
        gradient_global_matches: list = []
        gg_rej: dict = {}

        if gradient_enabled:
            grad_result = find_neighbor_via_gradient(
                c, depth, sobel_edges, workspace_mask, plane,
                z_visible_min=g.z_visible_min, config=config,
                obb_extent_xy_m=g.obb_extent_xy,
            )
            gradient_z = grad_result.z_highest_neighbor
            gradient_label = grad_result.chosen_label
            gradient_plateaus = grad_result.plateaus
            gradient_ring_px = grad_result.n_ring_pixels
            gradient_components = grad_result.n_components_total
            gradient_rej = grad_result.rejection_counts or {}
            plat_summary = ", ".join(
                f"{p.height_above_pallet:+.3f}m"
                f"(A={p.area_m2*1e4:.0f}cm²,σ={p.height_std_m*1000:.0f}mm,ar={p.aspect_ratio:.2f})"
                for p in sorted(
                    gradient_plateaus, key=lambda x: -x.height_above_pallet,
                )[:8]
            ) or "-"
            rej_summary = (
                f"area_px={gradient_rej.get('area_px', 0)}"
                f" ar={gradient_rej.get('aspect', 0)}"
                f" m2={gradient_rej.get('area_m2', 0)}"
                f" σ={gradient_rej.get('z_std', 0)}"
            )
            picked = "None" if gradient_z is None else f"{gradient_z:.3f}m"
            r_m = grad_result.effective_radius_m
            r_str = f"{r_m:.2f}m" if r_m == r_m else "n/a"  # NaN check
            print(
                f"[BOTTOM-GR] {c.candidate_id[:8]} z_min={g.z_visible_min:.3f}m "
                f"obb={g.obb_extent_xy:.2f}m r={r_str}/{grad_result.radius_px}px "
                f"ring={gradient_ring_px}px comp={gradient_components} "
                f"kept={len(gradient_plateaus)} rej[{rej_summary}] "
                f"plateaus=[{plat_summary}] -> picked={picked}"
            )
        else:
            gradient_z = None
            gradient_label = None
            gradient_plateaus = []
            gradient_ring_px = 0
            gradient_components = 0
            gradient_rej = {}

        if global_gradient_catalog:
            gg_result = find_neighbor_from_gradient_catalog(
                c, global_gradient_catalog, g.z_visible_min, config,
                depth=depth,
            )
            gradient_global_z = gg_result.z_highest_neighbor
            gradient_global_id = gg_result.chosen_global_id
            gradient_global_matches = gg_result.matching_plateaus
            gg_rej = gg_result.rejection_counts or {}
            match_summary = ", ".join(
                f"id{p.global_id}@{p.height_above_pallet:+.3f}m"
                for p in sorted(
                    gradient_global_matches,
                    key=lambda x: -x.height_above_pallet,
                )[:6]
            ) or "-"
            picked_gg = (
                "None" if gradient_global_z is None else f"{gradient_global_z:.3f}m"
            )
            print(
                f"[BOTTOM-GG] {c.candidate_id[:8]} z_min={g.z_visible_min:.3f}m "
                f"catalog={len(global_gradient_catalog)} "
                f"matched={len(gradient_global_matches)} "
                f"rej[high={gg_rej.get('too_high', 0)} "
                f"nooverlap={gg_rej.get('no_overlap', 0)}] "
                f"hits=[{match_summary}] -> picked={picked_gg}"
            )
        else:
            gradient_global_z = None
            gradient_global_id = None
            gradient_global_matches = []
            gg_rej = {}

        if histogram_enabled:
            hist_result = find_neighbor_via_depth_histogram(
                c, depth, workspace_mask, plane,
                z_visible_min=g.z_visible_min, config=config,
                obb_extent_xy_m=g.obb_extent_xy,
            )
            histogram_z = hist_result.z_highest_neighbor
            histogram_bands = hist_result.bands
            histogram_rej = hist_result.rejection_counts or {}
            band_summary = ", ".join(
                f"{b.height_median:+.3f}m"
                f"(A={b.area_m2*1e4:.0f}cm²,ar={b.aspect_ratio:.2f})"
                for b in sorted(histogram_bands, key=lambda x: -x.height_median)[:8]
            ) or "-"
            picked_h = "None" if histogram_z is None else f"{histogram_z:.3f}m"
            r_m_h = hist_result.effective_radius_m
            r_str_h = f"{r_m_h:.2f}m" if r_m_h == r_m_h else "n/a"
            print(
                f"[BOTTOM-HG] {c.candidate_id[:8]} z_min={g.z_visible_min:.3f}m "
                f"r={r_str_h}/{hist_result.radius_px}px "
                f"ring={hist_result.n_ring_pixels}px "
                f"bands_kept={len(histogram_bands)} "
                f"rej[low={histogram_rej.get('low_pixels', 0)} "
                f"comp={histogram_rej.get('no_component', 0)} "
                f"ar={histogram_rej.get('aspect', 0)} "
                f"m2={histogram_rej.get('area_m2', 0)}] "
                f"bands=[{band_summary}] -> picked={picked_h}"
            )
        else:
            histogram_z = None
            histogram_bands = []
            histogram_rej = {}

        if scene_planes:
            target_xy = np.asarray(c.points_3d, dtype=np.float64)[:, :2] \
                if c.points_3d is not None else np.zeros((0, 2))
            sp_result = find_neighbor_via_scene_planes(
                target_points_xy=target_xy,
                z_visible_min=g.z_visible_min,
                scene_planes=scene_planes,
                config=config,
            )
            scene_z = sp_result.z_highest_neighbor
            scene_id = sp_result.chosen_plane_id
            scene_qualifying = sp_result.qualifying_ids
            scene_rej = sp_result.rejection_counts
            qualifying_summary = ", ".join(
                f"id{pid}@{next((p.height_above_pallet for p in scene_planes if p.plane_id == pid), 0):+.3f}m"
                for pid in scene_qualifying[:6]
            ) or "-"
            picked_sp = "None" if scene_z is None else f"{scene_z:.3f}m"
            print(
                f"[BOTTOM-SP] {c.candidate_id[:8]} z_min={g.z_visible_min:.3f}m "
                f"rej[high={scene_rej.get('too_high', 0)} "
                f"nooverlap={scene_rej.get('no_overlap', 0)}] "
                f"qualifying=[{qualifying_summary}] -> picked={picked_sp}"
            )
        else:
            scene_z = None
            scene_id = None
            scene_qualifying = []
            scene_rej = {}

        lateral_info = find_lateral_neighbors(g, geom_index, config)

        if lateral_info.neighbor_ids:
            lat_summary = ", ".join(
                f"{nid[:6]}@{t:+.3f}m"
                for nid, t in sorted(
                    zip(lateral_info.neighbor_ids, lateral_info.neighbor_tops),
                    key=lambda x: -x[1],
                )[:6]
            )
            print(
                f"[BOTTOM-LT] {c.candidate_id[:8]} z_min={g.z_visible_min:.3f}m "
                f"r={lateral_info.effective_radius_m:.2f}m "
                f"others=[{lat_summary}] -> "
                f"highest={lateral_info.z_highest_neighbor:.3f}m"
            )

        z_highest_neighbor, highest_id, source = _resolve_neighbor_z(
            gradient_z, gradient_label,
            gradient_global_z, gradient_global_id,
            histogram_z, scene_z, scene_id, lateral_info,
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
        debug["z_neighbor_top_gradient_global"] = gradient_global_z
        debug["gradient_global_id_chosen"] = gradient_global_id
        debug["gradient_global_matched_ids"] = [
            p.global_id for p in gradient_global_matches if p.global_id is not None
        ]
        debug["gradient_global_n_matched"] = len(gradient_global_matches)
        debug["gradient_global_rejections"] = dict(gg_rej)
        debug["z_neighbor_top_histogram"] = histogram_z
        debug["z_neighbor_top_scene"] = scene_z
        debug["scene_plane_id_chosen"] = scene_id
        debug["scene_planes_qualifying"] = list(scene_qualifying)
        debug["scene_plane_rejections"] = dict(scene_rej)
        debug["z_neighbor_top_lateral"] = lateral_info.z_neighbor_top
        debug["z_highest_neighbor"] = z_highest_neighbor
        debug["neighbor_source"] = source
        debug["highest_neighbor_id"] = highest_id
        debug["case_label"] = decision.case_label
        debug["center_xy"] = g.center_xy.tolist()
        debug["obb_extent_xy"] = g.obb_extent_xy
        debug["z_pallet"] = z_pallet
        debug["gradient_n_ring_pixels"] = gradient_ring_px
        debug["gradient_n_components_total"] = gradient_components
        debug["gradient_n_plateaus_kept"] = len(gradient_plateaus)
        debug["gradient_rejections"] = dict(gradient_rej)
        debug["gradient_plateaus"] = [
            {
                "label": p.label,
                "area_px": p.area_px,
                "area_m2": p.area_m2,
                "height_above_pallet": p.height_above_pallet,
                "height_std_m": p.height_std_m,
                "aspect_ratio": p.aspect_ratio,
                "centroid_px": list(p.centroid_px),
            }
            for p in gradient_plateaus
        ]
        debug["histogram_n_bands_kept"] = len(histogram_bands)
        debug["histogram_rejections"] = dict(histogram_rej)
        debug["histogram_bands"] = [
            {
                "band_low": b.band_low,
                "band_high": b.band_high,
                "height_median": b.height_median,
                "area_px": b.area_px,
                "area_m2": b.area_m2,
                "aspect_ratio": b.aspect_ratio,
                "centroid_px": list(b.centroid_px),
            }
            for b in histogram_bands
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

    if return_scene_planes and return_gradient_catalog:
        return enriched, scene_planes, global_gradient_catalog
    if return_scene_planes:
        return enriched, scene_planes
    if return_gradient_catalog:
        return enriched, global_gradient_catalog
    return enriched
