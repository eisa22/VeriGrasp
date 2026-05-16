"""
Palettenebene (RANSAC) und horizontaler Workspace pro Session.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from config import (
    PALLET_MAX_NORMAL_ANGLE_DEG,
    PALLET_MIN_INLIER_RATIO,
    PALLET_RANSAC_DISTANCE_M,
    PALLET_RANSAC_ITERATIONS,
    PALLET_RANSAC_N,
    PALLET_RANSAC_STRIDE,
    PALLET_RANSAC_Y_MIN_RATIO,
    WORKSPACE_MIN_BOX_OVERLAP,
    WORKSPACE_X_MARGIN_LEFT,
    WORKSPACE_X_MARGIN_RIGHT,
)
from path_utils import get_depth_path

FX = FY = 437.04


@dataclass
class SessionContext:
    depth_abs: np.ndarray
    depth_rel: np.ndarray
    workspace_mask: np.ndarray
    plane_model: np.ndarray
    z_pallet_m: float
    x_range: tuple[int, int]


def build_workspace_mask(H: int, W: int) -> tuple[np.ndarray, tuple[int, int]]:
    """True im zentralen X-Bereich; links/rechts ausgeschlossen."""
    x_min = int(W * WORKSPACE_X_MARGIN_LEFT)
    x_max = int(W * (1.0 - WORKSPACE_X_MARGIN_RIGHT))
    x_min = max(0, min(x_min, W - 1))
    x_max = max(x_min + 1, min(x_max, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[:, x_min:x_max] = True
    return mask, (x_min, x_max)


def _build_ransac_mask(
    depth: np.ndarray,
    workspace_mask: np.ndarray,
) -> np.ndarray:
    H, W = depth.shape
    valid = (depth > 0) & workspace_mask
    if PALLET_RANSAC_Y_MIN_RATIO > 0:
        y_min = int(H * PALLET_RANSAC_Y_MIN_RATIO)
        valid &= np.arange(H)[:, None] >= y_min
    return valid


def _orient_plane_normal(plane_model: np.ndarray) -> np.ndarray:
    """Normalenrichtung: positive Z-Komponente (Kamera-Tiefenachse)."""
    a, b, c, d = plane_model
    if c < 0:
        return np.array([-a, -b, -c, -d], dtype=np.float64)
    return np.array([a, b, c, d], dtype=np.float64)


def _normal_angle_to_z_deg(plane_model: np.ndarray) -> float:
    a, b, c, _ = plane_model
    norm = np.sqrt(a * a + b * b + c * c)
    if norm < 1e-9:
        return 90.0
    cos_angle = abs(c) / norm
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def estimate_pallet_plane_ransac(
    depth: np.ndarray,
    workspace_mask: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """
    RANSAC-Ebenenschätzung auf der Palette (nur Workspace-Punkte).

    Returns:
        (plane_model [a,b,c,d], z_pallet_m, inlier_ratio) oder None
    """
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    ransac_mask = _build_ransac_mask(depth, workspace_mask)
    if ransac_mask.sum() < 500:
        print("[PALLET] Zu wenig Punkte für RANSAC.")
        return None

    stride = max(1, PALLET_RANSAC_STRIDE)
    ys, xs = np.where(ransac_mask)
    ys = ys[::stride]
    xs = xs[::stride]
    z = depth[ys, xs]
    x = (xs - cx) * z / FX
    y = (ys - cy) * z / FY
    points = np.stack([x, y, z], axis=1)

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PALLET_RANSAC_DISTANCE_M,
        ransac_n=PALLET_RANSAC_N,
        num_iterations=PALLET_RANSAC_ITERATIONS,
    )
    plane_model = _orient_plane_normal(np.asarray(plane_model, dtype=np.float64))
    inlier_ratio = len(inliers) / len(points) if len(points) > 0 else 0.0
    angle_deg = _normal_angle_to_z_deg(plane_model)

    if inlier_ratio < PALLET_MIN_INLIER_RATIO:
        print(
            f"[PALLET] RANSAC abgelehnt: inliers={inlier_ratio*100:.1f}% "
            f"< {PALLET_MIN_INLIER_RATIO*100:.0f}%"
        )
        return None
    if angle_deg > PALLET_MAX_NORMAL_ANGLE_DEG:
        print(
            f"[PALLET] RANSAC abgelehnt: Normalenwinkel {angle_deg:.1f}° "
            f"> {PALLET_MAX_NORMAL_ANGLE_DEG:.0f}°"
        )
        return None

    inlier_pts = points[inliers]
    z_pallet_m = float(np.median(inlier_pts[:, 2]))
    return plane_model, z_pallet_m, inlier_ratio


def depth_to_pallet_relative(
    depth: np.ndarray,
    plane_model: np.ndarray,
    workspace_mask: np.ndarray,
) -> np.ndarray:
    """
    Höhe über der Palettenebene (m). Positiv = näher zur Kamera / über der Palette.
    """
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float64)
    x = (u - cx) * z / FX
    y = (v - cy) * z / FY
    a, b, c, d = plane_model
    norm = np.sqrt(a * a + b * b + c * c)
    signed = (a * x + b * y + c * z + d) / norm
    # Näher zur Kamera als die Palette → positive Höhe
    depth_rel = -signed
    invalid = (depth <= 0) | (~workspace_mask)
    depth_rel = depth_rel.astype(np.float32)
    depth_rel[invalid] = 0.0
    return depth_rel


def get_working_depth(ctx: SessionContext) -> np.ndarray:
    """depth_rel mit Workspace-Maske (außerhalb = 0)."""
    depth_work = ctx.depth_rel.copy()
    depth_work[~ctx.workspace_mask] = 0.0
    return depth_work


def box_workspace_overlap(box, workspace_mask: np.ndarray) -> float:
    """Anteil der BBox-Fläche innerhalb des Workspace."""
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = workspace_mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = workspace_mask[y1:y2, x1:x2]
    return float(region.mean())


def filter_boxes_by_workspace(boxes, scores, labels, ctx: SessionContext):
    """Verwirft Boxen außerhalb des Workspace (Mittelpunkt oder zu wenig Überlappung)."""
    x_min, x_max = ctx.x_range
    kept_boxes, kept_scores, kept_labels = [], [], []
    removed = 0

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        overlap = box_workspace_overlap(box, ctx.workspace_mask)
        if cx < x_min or cx >= x_max or overlap < WORKSPACE_MIN_BOX_OVERLAP:
            removed += 1
            print(
                f"[WORKSPACE] Box '{label}' verworfen: "
                f"cx={cx:.0f} (x=[{x_min},{x_max}]), overlap={overlap*100:.0f}%"
            )
            continue
        kept_boxes.append(box)
        kept_scores.append(score)
        kept_labels.append(label)

    if removed > 0:
        print(f"[WORKSPACE] {removed} DINO-Box(en) außerhalb Workspace verworfen")
    return kept_boxes, kept_scores, kept_labels


def prepare_session_context(session_path: str) -> SessionContext | None:
    """Einmal pro Session: Workspace + RANSAC-Palettenebene."""
    depth_path = get_depth_path(session_path)
    if not os.path.exists(depth_path):
        print(f"[PALLET] Depth nicht gefunden: {depth_path}")
        return None

    depth_abs = np.load(depth_path).astype(np.float32)
    H, W = depth_abs.shape
    workspace_mask, x_range = build_workspace_mask(H, W)
    x_min, x_max = x_range

    print(
        f"[WORKSPACE] x=[{x_min},{x_max}] von {W}px "
        f"({WORKSPACE_X_MARGIN_LEFT*100:.0f}% links, "
        f"{WORKSPACE_X_MARGIN_RIGHT*100:.0f}% rechts ignoriert)"
    )

    result = estimate_pallet_plane_ransac(depth_abs, workspace_mask)
    if result is None:
        print("[PALLET] Fallback: keine Ebene – depth_rel = absolute Tiefe")
        depth_rel = depth_abs.copy()
        depth_rel[depth_abs <= 0] = 0.0
        depth_rel[~workspace_mask] = 0.0
        plane_model = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        z_pallet_m = float(np.median(depth_abs[(depth_abs > 0) & workspace_mask]))
    else:
        plane_model, z_pallet_m, inlier_ratio = result
        depth_rel = depth_to_pallet_relative(depth_abs, plane_model, workspace_mask)
        a, b, c, d = plane_model
        print(
            f"[PALLET] RANSAC: inliers={inlier_ratio*100:.1f}%, "
            f"z_pallet={z_pallet_m:.3f}m, normal=({a:.3f},{b:.3f},{c:.3f})"
        )

    return SessionContext(
        depth_abs=depth_abs,
        depth_rel=depth_rel,
        workspace_mask=workspace_mask,
        plane_model=plane_model,
        z_pallet_m=z_pallet_m,
        x_range=x_range,
    )
