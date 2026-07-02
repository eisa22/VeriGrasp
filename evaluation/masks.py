"""Mask I/O, workspace clipping, and IoU utilities."""

from __future__ import annotations

import numpy as np


def clip_mask(mask: np.ndarray, workspace_mask: np.ndarray) -> np.ndarray:
    """Apply workspace boolean mask; returns uint8 {0,1}."""
    m = (np.asarray(mask) > 0) & np.asarray(workspace_mask, dtype=bool)
    return m.astype(np.uint8)


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    """Shape (n_pred, n_gt)."""
    if not pred_masks or not gt_masks:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    out = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    for p, pm in enumerate(pred_masks):
        for g, gm in enumerate(gt_masks):
            out[p, g] = mask_iou(pm, gm)
    return out


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def encode_masks_rle(masks: list[np.ndarray]) -> np.ndarray:
    """Pack list of HxW masks into object array of run-length count arrays (Fortran order)."""
    packed = []
    for m in masks:
        m = (np.asarray(m) > 0).astype(np.uint8)
        flat = m.flatten(order="F")
        counts: list[int] = []
        prev = 0
        run = 0
        for v in flat:
            if v == prev:
                run += 1
            else:
                counts.append(run)
                run = 1
                prev = int(v)
        counts.append(run)
        packed.append(np.array(counts, dtype=np.int32))
    return np.array(packed, dtype=object)


def decode_masks_rle(
    rle_array: np.ndarray, height: int, width: int
) -> list[np.ndarray]:
    masks = []
    for counts in rle_array:
        flat = np.zeros(height * width, dtype=np.uint8)
        idx = 0
        val = 0
        for run in counts:
            flat[idx : idx + int(run)] = val
            idx += int(run)
            val = 1 - val
        masks.append(flat.reshape((height, width), order="F"))
    return masks
