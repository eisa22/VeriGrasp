"""SuctionNet normal_std backend (ported from suctionnet-baseline)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from perception.grasp_generation.camera import CameraInfo, depth_to_point_cloud
from perception.grasp_generation.centroid import build_centroid_disk_mask
from perception.grasp_generation.types import SuctionGrasp


def _orient_normals_to_direction(normals: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Open3D-Äquivalent in NumPy (orient_normals_to_align_with_direction crasht auf macOS)."""
    ref = np.asarray(direction, dtype=np.float64)
    ref_norm = np.linalg.norm(ref)
    if ref_norm <= 1e-12:
        return normals
    ref = ref / ref_norm
    oriented = normals.astype(np.float64, copy=True)
    flip = oriented @ ref < 0.0
    oriented[flip] *= -1.0
    return oriented.astype(np.float32, copy=False)


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def _std_filter(img: np.ndarray, wlen: int) -> np.ndarray:
    wmean, wsqrmean = (
        cv2.boxFilter(x, -1, (wlen, wlen), borderType=cv2.BORDER_REFLECT)
        for x in (img, img * img)
    )
    return np.sqrt(np.abs(wsqrmean - wmean * wmean))


def estimate_suction_heatmap(
    depth_img: np.ndarray,
    obj_mask: np.ndarray,
    camera: CameraInfo,
    normal_knn: int = 224,
    filter_size: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return heatmap, normal_map, organized point cloud."""
    point_cloud = depth_to_point_cloud(depth_img, camera)
    valid_idx = np.zeros_like(obj_mask, dtype=bool)
    coord1, coord2 = np.nonzero(obj_mask)
    if coord1.size == 0:
        h, w = depth_img.shape[:2]
        return (
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w, 3), dtype=np.float32),
            point_cloud,
        )
    coord1_min, coord1_max = int(coord1.min()), int(coord1.max())
    coord2_min, coord2_max = int(coord2.min()), int(coord2.max())
    valid_idx[coord1_min : coord1_max + 1, coord2_min : coord2_max + 1] = True
    valid_idx = valid_idx & (point_cloud[..., 2] != 0)

    height, width, _ = point_cloud.shape
    point_cloud_valid = point_cloud[valid_idx]
    if point_cloud_valid.shape[0] < 3:
        return (
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width, 3), dtype=np.float32),
            point_cloud,
        )

    pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud_valid))
    pc_o3d.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(normal_knn),
        fast_normal_computation=False,
    )
    normals = _normalize_normals(
        _orient_normals_to_direction(np.asarray(pc_o3d.normals), np.array([0.0, 0.0, -1.0]))
    )

    normal_map = np.zeros((height, width, 3), dtype=np.float32)
    normal_map[valid_idx] = normals
    mean_normal_std = np.mean(_std_filter(normal_map, filter_size), axis=2)
    max_std = float(np.max(mean_normal_std))
    if max_std <= 0:
        heatmap = np.zeros((height, width), dtype=np.float32)
    else:
        heatmap = (1.0 - mean_normal_std / max_std).astype(np.float32)
    heatmap[~valid_idx] = 0.0
    return heatmap, normal_map, point_cloud


def uniform_kernel(kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    return kernel / (kernel_size**2)


def smooth_heatmap(heatmap: np.ndarray, kernel_size: int) -> np.ndarray:
    k = kernel_size
    kernel = uniform_kernel(k)
    t = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
    padded = np.pad(heatmap, k // 2)
    h = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)
    return F.conv2d(h, t).squeeze().numpy()


def grid_sample(
    pred_score_map: np.ndarray,
    down_rate: int = 10,
    topk: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_row = pred_score_map.shape[0] // down_rate
    num_col = pred_score_map.shape[1] // down_rate
    idx_list = []
    for i in range(num_row):
        for j in range(num_col):
            pred_score_grid = pred_score_map[
                i * down_rate : (i + 1) * down_rate,
                j * down_rate : (j + 1) * down_rate,
            ]
            max_idx = np.argmax(pred_score_grid)
            max_idx = np.array([max_idx // down_rate, max_idx % down_rate], dtype=np.int32)
            max_idx[0] += i * down_rate
            max_idx[1] += j * down_rate
            idx_list.append(max_idx[np.newaxis, ...])
    if not idx_list:
        return np.array([]), np.array([]), np.array([])
    idx = np.concatenate(idx_list, axis=0)
    suction_scores = pred_score_map[idx[:, 0], idx[:, 1]]
    sort_idx = np.argsort(suction_scores)[::-1][:topk]
    return suction_scores[sort_idx], idx[:, 0][sort_idx], idx[:, 1][sort_idx]


def _apply_mask_filter(
    scores: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    mask: np.ndarray,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = []
    for i in range(len(scores)):
        r, c = int(rows[i]), int(cols[i])
        if r < 0 or c < 0 or r >= mask.shape[0] or c >= mask.shape[1]:
            continue
        if not mask[r, c]:
            continue
        if scores[i] < min_score:
            continue
        keep.append(i)
    if not keep:
        return np.array([]), np.array([]), np.array([])
    keep = np.asarray(keep, dtype=np.int64)
    return scores[keep], rows[keep], cols[keep]


def _apply_separation_filter(
    scores: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    point_cloud: np.ndarray,
    min_separation_m: float,
    max_grasps: int,
) -> list[tuple[float, int, int, np.ndarray, np.ndarray]]:
    """Greedy NMS in 3D by score."""
    order = np.argsort(scores)[::-1]
    selected: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    for i in order:
        r, c = int(rows[i]), int(cols[i])
        pos = point_cloud[r, c, :].astype(np.float64)
        if not np.all(np.isfinite(pos)) or pos[2] <= 0:
            continue
        too_close = False
        for _, _, _, prev_pos, _ in selected:
            if np.linalg.norm(pos - prev_pos) < min_separation_m:
                too_close = True
                break
        if too_close:
            continue
        selected.append((float(scores[i]), r, c, pos, point_cloud[r, c, :]))
        if len(selected) >= max_grasps:
            break
    return selected


def _apply_centroid_zone_to_heatmap(
    heatmap: np.ndarray,
    point_cloud: np.ndarray,
    anchor_3d: np.ndarray | None,
    radius_m: float | None,
    use_xy_distance: bool,
) -> tuple[np.ndarray, int]:
    if anchor_3d is None or radius_m is None:
        return heatmap, int(np.count_nonzero(heatmap > 0))
    disk = build_centroid_disk_mask(
        point_cloud, anchor_3d, radius_m, use_xy_distance=use_xy_distance
    )
    masked = heatmap * disk.astype(np.float32)
    return masked, int(np.count_nonzero(masked > 0))


def run_normal_std(
    depth_img: np.ndarray,
    obj_mask: np.ndarray,
    camera: CameraInfo,
    config: dict[str, Any],
    centroid_anchor: np.ndarray | None = None,
    centroid_radius_m: float | None = None,
    centroid_debug: dict[str, Any] | None = None,
) -> tuple[list[SuctionGrasp], dict[str, Any]]:
    heatmap, normal_map, point_cloud = estimate_suction_heatmap(
        depth_img,
        obj_mask,
        camera,
        normal_knn=int(config.get("normal_knn", 224)),
        filter_size=int(config.get("normal_std_filter_size", 25)),
    )
    k_size = int(config.get("heatmap_kernel_size", 15))
    heatmap_smooth = smooth_heatmap(heatmap, k_size)

    cc = config.get("centroid_constraint") or {}
    use_xy = bool(cc.get("use_xy_distance", True))
    radius_used = centroid_radius_m
    anchor_used = centroid_anchor
    heatmap, n_zone_px = _apply_centroid_zone_to_heatmap(
        heatmap_smooth.copy(),
        point_cloud,
        anchor_used,
        radius_used,
        use_xy,
    )
    relaxed = False
    if n_zone_px == 0 and cc.get("relax_on_empty", True) and anchor_used is not None:
        r_max = float(cc.get("max_radius_m", 0.15))
        if radius_used is not None and radius_used < r_max:
            radius_used = r_max
            relaxed = True
            heatmap, n_zone_px = _apply_centroid_zone_to_heatmap(
                heatmap_smooth.copy(),
                point_cloud,
                anchor_used,
                radius_used,
                use_xy,
            )

    scores, rows, cols = grid_sample(
        heatmap,
        down_rate=int(config.get("grid_down_rate", 10)),
        topk=int(config.get("grid_topk", 1024)),
    )
    scores, rows, cols = _apply_mask_filter(
        scores, rows, cols, obj_mask, float(config.get("min_score", 0.3))
    )

    selected = _apply_separation_filter(
        scores,
        rows,
        cols,
        point_cloud,
        float(config.get("min_separation_m", 0.04)),
        int(config.get("max_grasps", 20)),
    )

    grasps: list[SuctionGrasp] = []
    for rank, (score, r, c, _pos, _) in enumerate(selected):
        normal = normal_map[r, c, :].astype(np.float32)
        n_norm = np.linalg.norm(normal)
        if n_norm > 1e-6:
            normal = normal / n_norm
        grasps.append(
            SuctionGrasp(
                score=score,
                normal=normal,
                position=point_cloud[r, c, :].astype(np.float32),
                row=r,
                col=c,
                rank=rank,
            )
        )

    debug: dict[str, Any] = {
        "heatmap_max": float(np.max(heatmap)) if heatmap.size else 0.0,
        "heatmap_mean_in_mask": float(np.mean(heatmap[obj_mask])) if np.any(obj_mask) else 0.0,
        "candidates_after_mask": int(len(scores)),
        "n_selected": len(grasps),
        "heatmap_pixels_inside_zone": n_zone_px,
        "centroid_zone_relaxed": relaxed,
    }
    if centroid_debug:
        debug.update(centroid_debug)
    if radius_used is not None and radius_used != centroid_radius_m:
        debug["radius_m_relaxed"] = radius_used
    if len(grasps) == 0 and anchor_used is not None and n_zone_px == 0:
        debug["error"] = "centroid_zone_empty"
    return grasps, debug
