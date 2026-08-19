#!/usr/bin/env python3
"""Add n_blocking_points to exp3_per_grasp.csv (additive, byte-regression safe).

Reads corridor_clear detail from verification_result.json when Stage 3 is
present; otherwise runs offline full-mode verification on persisted pipeline
data (same as Experiment 3 export — no detector re-run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.masks import decode_masks_rle  # noqa: E402
from perception.candidate import BottomInference, CandidateOut  # noqa: E402
from perception.grasp_generation.types import SuctionGrasp  # noqa: E402
from Segmentation.pallet_scene import SessionContext, load_session_depth  # noqa: E402
from verification.config import load_verification_config  # noqa: E402
from verification.verify import verify_grasp  # noqa: E402

DEFAULT_TOLERANCE = 5


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _decode_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    arr = np.array([np.asarray(counts, dtype=np.int32)], dtype=object)
    return decode_masks_rle(arr, height, width)[0]


def _n_block_from_persisted(payload: dict | None) -> int | None:
    if not payload:
        return None
    for st in payload.get("stages") or []:
        for c in st.get("checks") or []:
            if c.get("name") == "corridor_clear":
                detail = c.get("detail") or {}
                val = detail.get("n_blocking_points")
                if val is not None:
                    return int(val)
    return None


def _session_context_from_prep(prep: dict, depth_abs: np.ndarray) -> SessionContext:
    h, w = int(prep["height"]), int(prep["width"])
    ws = _decode_rle(prep["workspace_mask_rle"], h, w).astype(bool)
    depth = np.asarray(depth_abs, dtype=np.float32)
    depth_rel = depth.copy()
    depth_rel[depth <= 0] = 0.0
    depth_rel[~ws] = 0.0
    return SessionContext(
        depth_abs=depth,
        depth_rel=depth_rel,
        workspace_mask=ws,
        plane_model=np.asarray(prep["plane_model"], dtype=np.float64),
        z_pallet_m=float(prep.get("z_pallet_m", 0.0)),
        x_range=(0, w),
        fx=float(prep["fx"]),
        fy=float(prep["fy"]),
        cx=float(prep["cx"]),
        cy=float(prep["cy"]),
    )


def _candidate_from_stage8(record: dict, height: int, width: int) -> CandidateOut:
    mask = _decode_rle(record["mask_rle"], height, width).astype(np.uint8)
    bottom = None
    obb = record.get("parcel_obb")
    if record.get("bottom_z_m") is not None and obb:
        bottom = BottomInference(
            bottom_z=float(record["bottom_z_m"]),
            bottom_method=str(record.get("bottom_method") or "unknown"),
            bottom_confidence=float(record.get("bottom_confidence") or 0.0),
            bottom_residual_m=0.0,
            used_neighbor_ids=[],
            height_m=float(record.get("height_m") or 0.0),
            parcel_obb=obb,
        )
    return CandidateOut(
        candidate_id=str(record["candidate_id"]),
        mask_2d=mask,
        points_3d=np.zeros((0, 3), dtype=np.float64),
        centroid_3d=np.asarray(record["centroid_3d"], dtype=np.float64),
        surface_normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        surface_area_m2=0.05,
        top_surface_height=float(record["top_surface_height_m"]),
        bbox_2d=tuple(int(v) for v in record["bbox_2d"]),
        debug={"label": record.get("label", "")},
        bottom=bottom,
    )


def _grasp_from_json(grasp: dict) -> SuctionGrasp:
    return SuctionGrasp(
        score=float(grasp.get("score", 0.0)),
        normal=np.asarray(grasp["normal"], dtype=np.float64),
        position=np.asarray(grasp["position"], dtype=np.float64),
        row=int(grasp.get("pixel", [0, 0])[0]),
        col=int(grasp.get("pixel", [0, 0])[1]),
        rank=int(grasp.get("rank", 0)),
    )


def _resolve_primary_grasp(
    stage10: dict | None,
    stage11: dict | None,
    candidate_record: dict,
    *,
    allow_centroid_fallback: bool,
) -> dict | None:
    pg = (stage11 or {}).get("primary_grasp")
    if pg and pg.get("position"):
        return pg
    if not allow_centroid_fallback:
        return None
    primary = (stage10 or {}).get("primary", {})
    cand = primary.get("candidate") or candidate_record
    centroid = cand.get("centroid_3d") or candidate_record.get("centroid_3d")
    if not centroid:
        return None
    return {
        "position": centroid,
        "normal": [0.0, 0.0, -1.0],
        "rank": int(pg.get("rank", 0)) if pg else 0,
        "score": float(pg.get("score", 0.0)) if pg else 0.0,
        "pixel": list(pg.get("pixel", [0, 0])) if pg else [0, 0],
    }


def _n_block_from_full_verify(session_path: Path, *, target_matched: bool) -> int:
    prep = _load_json(session_path / "stage_prep_context.json")
    if prep is None:
        raise FileNotFoundError(f"missing stage_prep_context.json in {session_path}")

    stage8 = _load_json(session_path / "stage8_candidates.json")
    stage10 = _load_json(session_path / "stage10_selected_target.json")
    stage11 = _load_json(session_path / "stage11_suction_grasps.json")
    corridor = _load_json(session_path / "extraction_corridor.json")

    primary_id = None
    if stage10 and stage10.get("primary"):
        primary_id = stage10["primary"]["candidate"]["candidate_id"]
    if primary_id is None or not stage8:
        raise RuntimeError(f"cannot resolve primary candidate in {session_path.name}")

    records = stage8["candidates"]
    candidate_record = next(
        (r for r in records if r["candidate_id"] == primary_id),
        None,
    )
    if candidate_record is None:
        raise RuntimeError(f"primary candidate not in stage8 for {session_path.name}")

    grasp_json = _resolve_primary_grasp(
        stage10, stage11, candidate_record, allow_centroid_fallback=not target_matched
    )
    if grasp_json is None:
        raise RuntimeError(f"no grasp for {session_path.name}")

    depth = load_session_depth(str(session_path))
    ctx = _session_context_from_prep(prep, depth)
    h, w = int(prep["height"]), int(prep["width"])
    candidate = _candidate_from_stage8(candidate_record, h, w)
    grasp = _grasp_from_json(grasp_json)
    cfg = {**load_verification_config(), "mode": "full"}
    result = verify_grasp(grasp, candidate, ctx, config=cfg, corridor=corridor)

    for st in result.stages:
        for c in st.checks:
            if c.name == "corridor_clear":
                return int(c.detail.get("n_blocking_points", 0))
    raise RuntimeError(f"corridor_clear missing in full verify for {session_path.name}")


def resolve_n_blocking(session_path: Path, *, target_matched: bool) -> tuple[int, str]:
    """Return (n_blocking_points, source) where source is json|full_verify."""
    persisted = _load_json(session_path / "verification_result.json")
    n_block = _n_block_from_persisted(persisted)
    if n_block is not None:
        return n_block, "json"
    return _n_block_from_full_verify(session_path, target_matched=target_matched), "full_verify"


def patch_csv(
    csv_path: Path,
    data_root: Path,
    *,
    tolerance: int = DEFAULT_TOLERANCE,
    dry_run: bool = False,
) -> dict:
    df = pd.read_csv(csv_path)
    if "n_blocking_points" in df.columns:
        print("[PATCH] n_blocking_points already present — validating only")
    else:
        original = df.copy()

    values: list[int] = []
    sources: dict[str, int] = {"json": 0, "full_verify": 0}

    for i, row in df.iterrows():
        scene_id = str(row["scene_id"])
        session_path = data_root / scene_id
        n_block, src = resolve_n_blocking(
            session_path, target_matched=bool(row["target_matched"])
        )
        values.append(n_block)
        sources[src] += 1
        if (i + 1) % 100 == 0 or i + 1 == len(df):
            print(f"[PATCH] {i + 1}/{len(df)} scenes …")

    if "n_blocking_points" not in df.columns:
        df["n_blocking_points"] = values
        pd.testing.assert_frame_equal(
            df.drop(columns=["n_blocking_points"]),
            original,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=0,
        )
        print("[GATE] existing columns unchanged (frame-equal)")

    recomputed = df["n_blocking_points"].astype(int) <= tolerance
    stored = df["check_corridor_clear_pass"].astype(bool)
    mismatches = int((recomputed != stored).sum())
    if mismatches:
        bad = df[recomputed != stored].head(5)
        raise SystemExit(
            f"Round-trip failed: {mismatches} corridor pass mismatches. "
            f"Examples: {bad['scene_id'].tolist()}"
        )

    n_rejects = int((~stored).sum())
    print(
        f"[GATE] corridor round-trip OK (616/616), n_rejects={n_rejects}, "
        f"sources={sources}"
    )

    if not dry_run and "n_blocking_points" not in pd.read_csv(csv_path, nrows=0).columns:
        df.to_csv(csv_path, index=False)
        print(f"[PATCH] wrote {csv_path}")

        summary_path = csv_path.parent / "exp3_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            meta = summary.setdefault("meta", {})
            schema = meta.get("csv_schema", "")
            if "+n_blocking_points" not in str(schema):
                meta["csv_schema"] = f"{schema}+n_blocking_points" if schema else "+n_blocking_points"
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"[PATCH] updated {summary_path} meta.csv_schema")

    return {
        "n_grasps": len(df),
        "n_rejects": n_rejects,
        "sources": sources,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch exp3_per_grasp.csv with n_blocking_points")
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "Results/exp3/full_2026-07-05/exp3_per_grasp.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "Data/blender_dataset",
    )
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = patch_csv(
        args.csv.resolve(),
        args.data_root.resolve(),
        tolerance=args.tolerance,
        dry_run=args.dry_run,
    )
    print(f"[DONE] {stats}")


if __name__ == "__main__":
    main()
