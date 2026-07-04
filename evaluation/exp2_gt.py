"""Experiment 2: GT derivations from the 8 bbox corners (pipeline pallet frame).

All quantities are derived from ``bbox_corners_camera_frame`` and expressed
relative to the pipeline's fitted pallet plane of the same scene (never the
dataset's true plane), so predicted and GT sides share one reference frame.
Upper/lower corners are identified by height above that plane, not by index
order, so tilted boxes are handled correctly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perception.geometry.plane import heights_above_plane, project_to_plane_xy


@dataclass
class GtObjectGeometry:
    """Pallet-frame geometry of one GT instance."""

    instance_id: int
    class_name: str
    visible_pixels: int
    center_camera: np.ndarray        # (3,) GT box centre, camera frame
    center_xy: np.ndarray            # (2,) centre projected into plane frame
    h_top: float                     # mean height of the 4 upper corners (m)
    h_bottom: float                  # mean height of the 4 lower corners (m)
    top_normal: np.ndarray           # (3,) unit, oriented toward the camera
    footprint_yaw_deg: float         # long-axis yaw in plane frame (deg)
    footprint_long_m: float
    footprint_short_m: float
    footprint_aspect: float          # long / short
    z_top_mean: float                # mean camera-frame z of the 4 upper corners

    @property
    def height_m(self) -> float:
        return self.h_top - self.h_bottom


def _box_edge_axes(corners: np.ndarray) -> list[np.ndarray]:
    """The three mutually orthogonal edge directions of a rectangular box.

    Reconstructed purely from the corner geometry (no index-order assumption):
    the vectors from one corner to the other seven are e1, e2, e3 and their
    pairwise/triple sums; picking the shortest mutually orthogonal ones
    recovers the edges. Returns fewer than 3 axes for degenerate input.
    """
    v = corners[1:] - corners[0]
    order = np.argsort(np.linalg.norm(v, axis=1))
    axes: list[np.ndarray] = []
    for idx in order:
        cand = v[idx]
        n_cand = np.linalg.norm(cand)
        if n_cand < 1e-9:
            continue
        ok = all(
            abs(float(cand @ a)) / (n_cand * np.linalg.norm(a)) < 0.2
            for a in axes
        )
        if ok:
            axes.append(cand)
        if len(axes) == 3:
            break
    return axes


def _top_face_normal(
    corners: np.ndarray,
    top_corners: np.ndarray,
    plane: tuple[float, float, float, float],
) -> np.ndarray:
    """Outward normal of the box face that points closest to pallet-up.

    Face normals of a rectangular box are (+/-) its three edge directions;
    the top face is the one whose outward normal is best aligned with the
    up direction of the pipeline's pallet plane. This is robust against
    canonical-frame variants and against tilted boxes where the four highest
    corners do not form a face (which breaks a naive plane fit).
    """
    from perception.geometry.plane import plane_basis

    n_plane, _, _ = plane_basis(plane)
    u_up = -n_plane  # heights_above_plane increases along -n_plane

    axes = _box_edge_axes(corners)
    if len(axes) == 3:
        best: np.ndarray | None = None
        best_dot = -1.0
        for a in axes:
            n = a / (np.linalg.norm(a) + 1e-12)
            d = float(n @ u_up)
            if abs(d) > best_dot:
                best_dot = abs(d)
                best = n if d >= 0 else -n
        # Aligned with pallet-up implies camera-facing (u_up has negative z).
        return best

    # Degenerate corners: fall back to a plane fit through the 4 highest.
    centered = top_corners - top_corners.mean(axis=0)
    _, _, vt = np.linalg.svd(centered)
    n = vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    if float(n @ u_up) < 0:
        n = -n
    return n


def _footprint_rect(corners_xy: np.ndarray) -> tuple[float, float, float]:
    """Min-area rectangle of the projected corners.

    Returns (yaw_deg of the long axis in plane frame, long extent, short extent).
    """
    import cv2

    pts = corners_xy.astype(np.float32).reshape(-1, 1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    # cv2 angle refers to the `w` side; make yaw refer to the long axis.
    if w >= h:
        long_e, short_e = float(w), float(h)
        yaw = float(angle)
    else:
        long_e, short_e = float(h), float(w)
        yaw = float(angle) + 90.0
    # Normalise into (-90, 90] — orientation is 180deg-symmetric anyway.
    yaw = ((yaw + 90.0) % 180.0) - 90.0
    return yaw, long_e, short_e


def derive_gt_geometry(
    obj: dict,
    plane: tuple[float, float, float, float],
) -> GtObjectGeometry:
    """Derive pallet-frame quantities for one GT object record."""
    corners = np.asarray(obj["bbox_corners_camera_frame"], dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"object {obj.get('id')}: expected (8,3) corners, got {corners.shape}")

    heights = heights_above_plane(corners, plane)
    order = np.argsort(-heights)          # descending: highest first
    top_idx, bottom_idx = order[:4], order[4:]

    # Spec (sec 4): h_gt_top/bottom from the 4 upper/lower corners by height.
    # The normal, by contrast, is a face property and comes from the box edges.
    h_top = float(np.mean(heights[top_idx]))
    h_bottom = float(np.mean(heights[bottom_idx]))
    top_corners = corners[top_idx]
    normal = _top_face_normal(corners, top_corners, plane)

    corners_xy = project_to_plane_xy(corners, plane)
    yaw_deg, long_e, short_e = _footprint_rect(corners_xy)

    center = np.asarray(obj["center_camera_frame"], dtype=np.float64)
    center_xy = project_to_plane_xy(center.reshape(1, 3), plane)[0]

    return GtObjectGeometry(
        instance_id=int(obj["id"]),
        class_name=str(obj.get("class_name", "unknown")),
        visible_pixels=int(obj.get("visible_pixels", 0)),
        center_camera=center,
        center_xy=center_xy,
        h_top=h_top,
        h_bottom=h_bottom,
        top_normal=normal,
        footprint_yaw_deg=yaw_deg,
        footprint_long_m=long_e,
        footprint_short_m=max(short_e, 1e-9),
        footprint_aspect=float(long_e / max(short_e, 1e-9)),
        z_top_mean=float(np.mean(corners[top_idx][:, 2])),
    )


def load_gt_geometries(
    session_path: str | Path,
    plane: tuple[float, float, float, float],
) -> dict[int, GtObjectGeometry]:
    """All GT objects of a scene keyed by instance id (ids >= 0 only)."""
    with open(Path(session_path) / "ground_truth.json", encoding="utf-8") as f:
        gt_json = json.load(f)
    out: dict[int, GtObjectGeometry] = {}
    for obj in gt_json.get("objects", []):
        if int(obj["id"]) < 0:
            continue
        out[int(obj["id"])] = derive_gt_geometry(obj, plane)
    return out


def visibility_ratio(
    geo: GtObjectGeometry,
    fx: float,
    fy: float,
) -> float:
    """Visible pixel count over the expected full top-face pixel area.

    The expected area projects the GT footprint rectangle at the depth of the
    top face; for the top-down camera this approximates the unoccluded
    visible-surface size well. Values can slightly exceed 1 due to the
    rectangle approximation.
    """
    z = max(geo.z_top_mean, 1e-6)
    expected_px = geo.footprint_long_m * geo.footprint_short_m * fx * fy / (z * z)
    if expected_px <= 0:
        return 0.0
    return float(geo.visible_pixels / expected_px)
