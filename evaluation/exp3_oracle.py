"""Experiment 3: grasp-validity oracle from GT geometry only.

This module must remain independent of measured depth and verification checks.
It reuses GT derivations from Experiment 2 and plane projection helpers only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evaluation.exp2_geometry import angle_between_deg, orient_toward_camera
from evaluation.exp2_gt import GtObjectGeometry
from perception.geometry.plane import (
    heights_above_plane,
    plane_basis,
    project_to_plane_xy,
    unproject_from_plane_xy,
)

ORACLE_CRITERIA = (
    "single_object",
    "footprint_containment",
    "surface_contact",
    "normal_alignment",
    "free_extraction",
)


@dataclass
class OracleParams:
    tol_contain_m: float = 0.005
    tol_contact_m: float = 0.020
    tol_normal_deg: float = 15.0
    gripper_half_long_m: float = 0.060
    gripper_half_short_m: float = 0.030
    corridor_height_m: float = 0.30


@dataclass
class OracleLabel:
    valid: bool
    violated: list[str] = field(default_factory=list)


def oracle_params_from_config(cfg: dict) -> OracleParams:
    w = float(cfg.get("gripper_width_m", 0.120))
    l = float(cfg.get("gripper_length_m", 0.060))
    return OracleParams(
        tol_contain_m=float(cfg.get("tol_contain_mm", 5)) / 1000.0,
        tol_contact_m=float(cfg.get("tol_contact_mm", 20)) / 1000.0,
        tol_normal_deg=float(cfg.get("tol_normal_deg", 15)),
        gripper_half_long_m=0.5 * max(w, l),
        gripper_half_short_m=0.5 * min(w, l),
        corridor_height_m=float(cfg.get("corridor_height_m", 0.30)),
    )


def _gripper_rot(long_dir_xy: np.ndarray | None) -> np.ndarray:
    if long_dir_xy is None:
        return np.eye(2, dtype=np.float64)
    d = np.asarray(long_dir_xy, dtype=np.float64).reshape(2)
    d = d / (np.linalg.norm(d) + 1e-12)
    return np.array([[d[0], -d[1]], [d[1], d[0]]], dtype=np.float64)


def yaw_to_long_dir_xy(yaw_deg: float, plane: tuple[float, float, float, float]) -> np.ndarray:
    """Unit long-axis direction in the pallet (u, v) frame."""
    _, u, v = plane_basis(plane)
    yaw = np.radians(float(yaw_deg))
    d = np.cos(yaw) * u + np.sin(yaw) * v
    d_xy = np.array([float(d @ u), float(d @ v)], dtype=np.float64)
    return d_xy / (np.linalg.norm(d_xy) + 1e-12)


def gripper_footprint_corners_plane(
    grasp_position: np.ndarray,
    plane: tuple[float, float, float, float],
    params: OracleParams,
    long_dir_xy: np.ndarray | None,
) -> np.ndarray:
    """Four footprint corners (4, 2) in the pallet plane frame."""
    g_xy = project_to_plane_xy(
        np.asarray(grasp_position, dtype=np.float64).reshape(1, 3), plane
    )[0]
    corners_g = np.array(
        [
            [-params.gripper_half_long_m, -params.gripper_half_short_m],
            [params.gripper_half_long_m, -params.gripper_half_short_m],
            [params.gripper_half_long_m, params.gripper_half_short_m],
            [-params.gripper_half_long_m, params.gripper_half_short_m],
        ],
        dtype=np.float64,
    )
    return g_xy.reshape(1, 2) + corners_g @ _gripper_rot(long_dir_xy).T


def gt_top_face_corners_plane(
    corners_3d: np.ndarray,
    plane: tuple[float, float, float, float],
) -> np.ndarray:
    """Project the four upper GT bbox corners into the pallet plane."""
    corners = np.asarray(corners_3d, dtype=np.float64)
    heights = heights_above_plane(corners, plane)
    top_idx = np.argsort(-heights)[:4]
    return project_to_plane_xy(corners[top_idx], plane)


def _point_in_convex_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """Cross-product sign test for a convex polygon (CCW or CW)."""
    p = np.asarray(point, dtype=np.float64).reshape(2)
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    o1 = cross(p1, p2, q1)
    o2 = cross(p1, p2, q2)
    o3 = cross(q1, q2, p1)
    o4 = cross(q1, q2, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return False


def polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Overlap test for two convex quadrilaterals in 2D."""
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    for poly, other in ((a, b), (b, a)):
        for p in poly:
            if _point_in_convex_polygon(p, other):
                return True
    for i in range(len(a)):
        for j in range(len(b)):
            if _segments_intersect(a[i], a[(i + 1) % len(a)], b[j], b[(j + 1) % len(b)]):
                return True
    return False


def _gt_top_face_rect(
    top_corners_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Centre, long/short axes (unit), and extents of the GT top-face rectangle."""
    import cv2

    pts = np.asarray(top_corners_xy, dtype=np.float32).reshape(-1, 1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if w >= h:
        long_e, short_e = float(w), float(h)
        rot = np.radians(float(angle))
    else:
        long_e, short_e = float(h), float(w)
        rot = np.radians(float(angle) + 90.0)
    centre = np.array([cx, cy], dtype=np.float64)
    long_u = np.array([np.cos(rot), np.sin(rot)], dtype=np.float64)
    short_u = np.array([-long_u[1], long_u[0]], dtype=np.float64)
    return centre, long_u, short_u, long_e, short_e


def footprint_inside_eroded_gt(
    gripper_corners: np.ndarray,
    gt_top_corners_xy: np.ndarray,
    tol_m: float,
) -> bool:
    centre, long_u, short_u, long_e, short_e = _gt_top_face_rect(gt_top_corners_xy)
    half_long = max(long_e * 0.5 - tol_m, 0.0)
    half_short = max(short_e * 0.5 - tol_m, 0.0)
    for corner in gripper_corners:
        d = np.asarray(corner, dtype=np.float64) - centre
        along = abs(float(d @ long_u))
        across = abs(float(d @ short_u))
        if along > half_long + 1e-9 or across > half_short + 1e-9:
            return False
    return True


def distance_to_top_plane_along_approach(
    grasp_position: np.ndarray,
    top_corners: np.ndarray,
    approach_normal: np.ndarray,
) -> float:
    """Signed distance from grasp to the GT top plane along the approach axis."""
    top = np.asarray(top_corners, dtype=np.float64)
    p0 = top.mean(axis=0)
    n_face = orient_toward_camera(
        np.cross(top[1] - top[0], top[2] - top[0])
    )
    n_app = orient_toward_camera(np.asarray(approach_normal, dtype=np.float64))
    diff = np.asarray(grasp_position, dtype=np.float64) - p0
    # Component along approach; use face normal sign for plane distance.
    plane_dist = abs(float(diff @ n_face))
    if abs(float(n_app @ n_face)) > 0.9:
        return plane_dist
    return abs(float(diff @ n_app))


def _prism_vertices_from_footprint(
    footprint_xy: np.ndarray,
    plane: tuple[float, float, float, float],
    h_bottom: float,
    h_top: float,
) -> np.ndarray:
    """Extrude a planar footprint between two heights above the pallet."""
    fp = np.asarray(footprint_xy, dtype=np.float64)
    bot = unproject_from_plane_xy(fp, plane, heights=h_bottom)
    top = unproject_from_plane_xy(fp, plane, heights=h_top)
    return np.vstack([bot, top])


def _project_interval(values: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    proj = values @ axis
    return float(proj.min()), float(proj.max())


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float], eps: float = 1e-9) -> bool:
    return a[1] >= b[0] - eps and b[1] >= a[0] - eps


def convex_polyhedra_intersect(verts_a: np.ndarray, verts_b: np.ndarray) -> bool:
    """SAT test for two convex polyhedra given by their vertices."""
    a = np.asarray(verts_a, dtype=np.float64)
    b = np.asarray(verts_b, dtype=np.float64)

    def face_normals(verts: np.ndarray) -> list[np.ndarray]:
        norms: list[np.ndarray] = []
        n = len(verts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    nrm = np.cross(verts[j] - verts[i], verts[k] - verts[i])
                    norm = np.linalg.norm(nrm)
                    if norm > 1e-9:
                        norms.append(nrm / norm)
        # Deduplicate roughly parallel normals.
        uniq: list[np.ndarray] = []
        for nrm in norms:
            if not any(abs(float(nrm @ u)) > 0.99 for u in uniq):
                uniq.append(nrm)
        return uniq

    axes: list[np.ndarray] = []
    axes.extend(face_normals(a))
    axes.extend(face_normals(b))
    edges_a = [a[(i + 1) % len(a)] - a[i] for i in range(len(a))]
    edges_b = [b[(i + 1) % len(b)] - b[i] for i in range(len(b))]
    for ea in edges_a:
        for eb in edges_b:
            ax = np.cross(ea, eb)
            norm = np.linalg.norm(ax)
            if norm > 1e-9:
                axes.append(ax / norm)

    for axis in axes:
        ia = _project_interval(a, axis)
        ib = _project_interval(b, axis)
        if not _intervals_overlap(ia, ib):
            return False
    return True


def _check_single_object(
    matched: GtObjectGeometry,
    gripper_corners: np.ndarray,
    all_gt_geos: dict[int, GtObjectGeometry],
    all_corners: dict[int, np.ndarray],
    plane: tuple[float, float, float, float],
    params: OracleParams,
) -> bool:
    for iid, geo in all_gt_geos.items():
        if iid == matched.instance_id:
            continue
        if abs(geo.h_top - matched.h_top) > params.tol_contact_m:
            continue
        other_fp = gt_top_face_corners_plane(all_corners[iid], plane)
        if polygons_overlap(gripper_corners, other_fp):
            return False
    return True


def _check_free_extraction(
    matched: GtObjectGeometry,
    matched_corners: np.ndarray,
    all_corners: dict[int, np.ndarray],
    plane: tuple[float, float, float, float],
    params: OracleParams,
) -> bool:
    fp = gt_top_face_corners_plane(matched_corners, plane)
    h0 = matched.h_top
    h1 = h0 + params.corridor_height_m
    prism = _prism_vertices_from_footprint(fp, plane, h0, h1)
    for iid, corners in all_corners.items():
        if iid == matched.instance_id:
            continue
        box = np.asarray(corners, dtype=np.float64)
        if convex_polyhedra_intersect(prism, box):
            return False
    return True


def evaluate_oracle(
    *,
    grasp_position: np.ndarray,
    grasp_normal: np.ndarray,
    pred_yaw_deg: float | None,
    target_matched: bool,
    matched_gt: GtObjectGeometry | None,
    matched_corners: np.ndarray | None,
    all_gt_geos: dict[int, GtObjectGeometry],
    all_corners: dict[int, np.ndarray],
    plane: tuple[float, float, float, float],
    params: OracleParams,
) -> OracleLabel:
    """Evaluate all five oracle criteria for one primary grasp."""
    if not target_matched or matched_gt is None or matched_corners is None:
        return OracleLabel(valid=False, violated=["single_object"])

    long_dir = (
        yaw_to_long_dir_xy(pred_yaw_deg, plane)
        if pred_yaw_deg is not None
        else None
    )
    gripper_fp = gripper_footprint_corners_plane(
        grasp_position, plane, params, long_dir
    )
    gt_fp = gt_top_face_corners_plane(matched_corners, plane)
    top_idx = np.argsort(-heights_above_plane(matched_corners, plane))[:4]
    top_corners_3d = np.asarray(matched_corners, dtype=np.float64)[top_idx]

    violated: list[str] = []

    if not _check_single_object(
        matched_gt, gripper_fp, all_gt_geos, all_corners, plane, params
    ):
        violated.append("single_object")

    if not footprint_inside_eroded_gt(gripper_fp, gt_fp, params.tol_contain_m):
        violated.append("footprint_containment")

    dist = distance_to_top_plane_along_approach(
        grasp_position, top_corners_3d, grasp_normal
    )
    if dist > params.tol_contact_m:
        violated.append("surface_contact")

    n_pred = orient_toward_camera(np.asarray(grasp_normal, dtype=np.float64))
    n_gt = orient_toward_camera(matched_gt.top_normal)
    if angle_between_deg(n_pred, n_gt) > params.tol_normal_deg:
        violated.append("normal_alignment")

    if not _check_free_extraction(
        matched_gt, matched_corners, all_corners, plane, params
    ):
        violated.append("free_extraction")

    return OracleLabel(valid=len(violated) == 0, violated=violated)


def load_gt_corner_boxes(session_path) -> dict[int, np.ndarray]:
    """Load 8-corner GT boxes keyed by instance id (ids >= 0 only)."""
    import json
    from pathlib import Path

    with open(Path(session_path) / "ground_truth.json", encoding="utf-8") as f:
        gt_json = json.load(f)
    out: dict[int, np.ndarray] = {}
    for obj in gt_json.get("objects", []):
        iid = int(obj["id"])
        if iid < 0:
            continue
        out[iid] = np.asarray(obj["bbox_corners_camera_frame"], dtype=np.float64)
    return out
