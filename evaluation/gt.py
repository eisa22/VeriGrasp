"""Ground-truth loading and evaluability filtering for Experiment 1."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evaluation.masks import clip_mask, mask_area, tight_bbox


@dataclass
class VisibilityFilter:
    """Controls which GT instances enter the recall denominator."""

    mode: str = "absolute"  # "absolute" | "relative_median"
    absolute_min: int = 1
    relative_fraction: float = 0.01

    @classmethod
    def from_legacy_min_visible(cls, min_visible_pixels: int) -> VisibilityFilter:
        return cls(mode="absolute", absolute_min=int(min_visible_pixels))


@dataclass
class GtInstance:
    instance_id: int
    class_name: str
    mask: np.ndarray
    visible_pixels: int
    visible_pixels_json: int | None
    bbox: tuple[int, int, int, int] | None
    evaluable: bool
    exclusion_reason: str | None  # "invisible" | "out_of_workspace" | None


@dataclass
class GtScene:
    scene_id: str
    instances: list[GtInstance] = field(default_factory=list)
    gt_eval: list[GtInstance] = field(default_factory=list)
    gt_invisible: int = 0
    gt_out_of_scope: int = 0
    workspace_mask: np.ndarray | None = None
    visibility_threshold_px: int | None = None

    @property
    def gt_total(self) -> int:
        return len(self.instances)


def load_instance_mask(session_path: str | Path) -> np.ndarray:
    path = Path(session_path) / "instance_mask.npy"
    return np.load(path)


def _class_map(gt_json: dict) -> dict[int, str]:
    out: dict[int, str] = {}
    for obj in gt_json.get("objects", []):
        out[int(obj["id"])] = str(obj.get("class_name", "unknown"))
    return out


def _json_visible_map(gt_json: dict) -> dict[int, int]:
    out: dict[int, int] = {}
    for obj in gt_json.get("objects", []):
        if "visible_pixels" in obj:
            out[int(obj["id"])] = int(obj["visible_pixels"])
    return out


def _object_ids_from_json(gt_json: dict) -> list[int]:
    return sorted(
        int(obj["id"])
        for obj in gt_json.get("objects", [])
        if int(obj["id"]) >= 0
    )


def _visible_pixel_count(
    lid: int,
    inst: np.ndarray,
    json_vis: dict[int, int],
) -> int:
    if lid in json_vis:
        return int(json_vis[lid])
    return mask_area((inst == lid).astype(np.uint8))


def _visibility_threshold_px(
    visibility: VisibilityFilter,
    workspace_visible_pixels: list[int],
) -> int:
    if visibility.mode == "relative_median":
        pool = [int(v) for v in workspace_visible_pixels if v > 0]
        if not pool:
            return max(visibility.absolute_min, 1)
        median_vis = float(np.median(pool))
        return max(1, int(np.ceil(visibility.relative_fraction * median_vis)))
    return max(int(visibility.absolute_min), 1)


def assert_gt_reconciliation(scene: GtScene, expected_total: int | None = None) -> None:
    """Every annotated instance must land in exactly one GT bucket."""
    total = expected_total if expected_total is not None else scene.gt_total
    accounted = scene.gt_eval.__len__() + scene.gt_invisible + scene.gt_out_of_scope
    assert accounted == total, (
        f"{scene.scene_id}: gt_eval({len(scene.gt_eval)}) + "
        f"invisible({scene.gt_invisible}) + out_of_scope({scene.gt_out_of_scope}) "
        f"= {accounted} != total({total})"
    )


def build_gt_scene(
    session_path: str | Path,
    workspace_mask: np.ndarray,
    *,
    min_visible_pixels: int = 1,
    visibility: VisibilityFilter | None = None,
    workspace_majority_threshold: float = 0.5,
    assert_reconciliation: bool = True,
) -> GtScene:
    session_path = Path(session_path)
    scene_id = session_path.name
    inst = load_instance_mask(session_path)
    with open(session_path / "ground_truth.json", encoding="utf-8") as f:
        gt_json = json.load(f)

    vis_filter = visibility or VisibilityFilter.from_legacy_min_visible(min_visible_pixels)
    classes = _class_map(gt_json)
    json_vis = _json_visible_map(gt_json)
    object_ids = _object_ids_from_json(gt_json)

    scene = GtScene(scene_id=scene_id, workspace_mask=workspace_mask)

    pending: list[dict] = []
    for lid in object_ids:
        raw = (inst == lid).astype(np.uint8)
        vis = _visible_pixel_count(lid, inst, json_vis)
        jvis = json_vis.get(lid)
        clipped = clip_mask(raw, workspace_mask)
        ws_px = mask_area(clipped)
        total_px = mask_area(raw)
        pending.append(
            {
                "lid": lid,
                "raw": raw,
                "vis": vis,
                "jvis": jvis,
                "clipped": clipped,
                "ws_px": ws_px,
                "total_px": total_px,
            }
        )

    workspace_pool = [
        row["vis"]
        for row in pending
        if row["total_px"] > 0
        and (row["ws_px"] / row["total_px"]) >= workspace_majority_threshold
    ]
    vis_threshold = _visibility_threshold_px(vis_filter, workspace_pool)
    scene.visibility_threshold_px = vis_threshold

    for row in pending:
        lid = row["lid"]
        vis = row["vis"]
        total_px = row["total_px"]
        ws_px = row["ws_px"]
        clipped = row["clipped"]
        raw = row["raw"]

        reason = None
        evaluable = True
        if total_px == 0 or vis < vis_threshold:
            evaluable = False
            reason = "invisible"
            scene.gt_invisible += 1
        elif total_px > 0 and (ws_px / total_px) < workspace_majority_threshold:
            evaluable = False
            reason = "out_of_workspace"
            scene.gt_out_of_scope += 1

        bbox = tight_bbox(clipped if evaluable else raw)
        gi = GtInstance(
            instance_id=lid,
            class_name=classes.get(lid, "unknown"),
            mask=clipped if evaluable else raw,
            visible_pixels=vis,
            visible_pixels_json=row["jvis"],
            bbox=bbox,
            evaluable=evaluable,
            exclusion_reason=reason,
        )
        scene.instances.append(gi)
        if evaluable:
            scene.gt_eval.append(gi)

    if assert_reconciliation:
        assert_gt_reconciliation(scene, expected_total=len(object_ids))

    return scene
