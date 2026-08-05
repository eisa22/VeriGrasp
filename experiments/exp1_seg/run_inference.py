"""Phase A: run perception pipeline and dump masks for Experiment 1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp1_seg._runtime import require_torch  # noqa: E402

require_torch()

import numpy as np
import torch
from transformers import (
    AutoModelForZeroShotObjectDetection as DinoModel,
    AutoProcessor as DinoProcessor,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from config import DINO_MODEL_ID  # noqa: E402
from evaluation.env_snapshot import (  # noqa: E402
    load_pipeline_config,
    write_config_snapshot,
    write_env_snapshot,
)
from evaluation.masks import encode_masks_rle  # noqa: E402
from experiments.exp1_seg.test_set import (  # noqa: E402
    filter_session_paths,
    list_test_set_names,
    test_set_manifest,
)
from path_utils import get_all_session_paths, get_data_root  # noqa: E402
from perception.pipeline import PerceptionArtifacts, run_perception  # noqa: E402


def _load_eval_config() -> dict:
    import yaml

    path = Path(__file__).parent / "eval_config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_dir(
    base: Path | None = None, test_set: str | None = None, variant: str = "baseline"
) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = base or (PROJECT_ROOT / "Results" / "exp1_seg")
    name = stamp if not test_set else f"{stamp}_test-{test_set}"
    if variant != "baseline":
        name = f"{name}_variant-{variant}"
    return root / name


def save_artifacts(run_dir: Path, scene_id: str, art: PerceptionArtifacts) -> Path:
    preds = run_dir / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    path = preds / f"{scene_id}.npz"
    h = art.height or 480
    w = art.width or 640
    ws = art.workspace_mask if art.workspace_mask is not None else np.ones((h, w), dtype=bool)

    np.savez_compressed(
        path,
        status=art.status,
        height=h,
        width=w,
        workspace_mask=ws.astype(np.uint8),
        boxes_D=np.array(art.boxes_D, dtype=np.int32) if art.boxes_D else np.zeros((0, 4), np.int32),
        scores_D=np.array(art.scores_D, dtype=np.float32),
        labels_D=np.array(art.labels_D, dtype=object),
        masks_S_rle=encode_masks_rle(art.masks_S),
        scores_S=np.array(art.scores_S, dtype=np.float32),
        labels_S=np.array(art.labels_S, dtype=object),
        masks_M_rle=encode_masks_rle(art.masks_M),
        scores_M=np.array(art.scores_M, dtype=np.float32),
        labels_M=np.array(art.labels_M, dtype=object),
        masks_F_rle=encode_masks_rle(art.masks_F),
        scores_F=np.array(art.scores_F, dtype=np.float32),
        labels_F=np.array(art.labels_F, dtype=object),
    )
    return path


def main() -> None:
    eval_cfg = _load_eval_config()
    test_names = list_test_set_names(eval_cfg)

    parser = argparse.ArgumentParser(description="Experiment 1 — Phase A inference + mask dump")
    parser.add_argument("--limit", type=int, default=None, help="Process first N scenes only")
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        choices=test_names if test_names else None,
        metavar="NAME",
        help=f"Named subset from eval_config.yaml: {', '.join(test_names)}",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open 3D perception visualisation per scene (Open3D, perception stages only)",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing or new run directory")
    parser.add_argument("--resume", action="store_true", help="Skip scenes with existing .npz")
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=["baseline", "sam3d"],
        help=(
            "Perception-Variante: 'baseline' = DINO→Sobel→Matching→SAM3D (Standard), "
            "'sam3d' = DINO→SAM→Dedup→SAM3D (SAM-Masken statt Sobel/Matching)"
        ),
    )
    args = parser.parse_args()

    if args.gui and args.limit is None and args.test_set is None:
        print("[EXP1] --gui without --test-set/--limit: using test-set 'smoke' (5 scenes)")
        args.test_set = "smoke"

    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EXP1] Device: {device}")
    if args.gui:
        print("[EXP1] GUI mode: Open3D windows per scene (close window to continue)")

    pipe_cfg = load_pipeline_config()
    config_blob = {"pipeline": pipe_cfg, "eval": eval_cfg}

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else _run_dir(test_set=args.test_set, variant=args.variant)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_env_snapshot(run_dir, device, config_blob)
    write_config_snapshot(run_dir, config_blob)
    print(f"[EXP1] Run directory: {run_dir}")

    if args.data_root:
        import config as cfg

        cfg.BASE_PATH = args.data_root

    sessions = get_all_session_paths()
    if args.test_set:
        sessions = filter_session_paths(sessions, args.test_set, eval_cfg)
        manifest = test_set_manifest(
            args.test_set, eval_cfg, [Path(p).name for p in sessions]
        )
        with open(run_dir / "test_set.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[EXP1] Test set '{args.test_set}': {manifest['n_scenes']} scenes")
    if args.limit:
        sessions = sessions[: args.limit]

    print(f"[EXP1] Loading DINO ({DINO_MODEL_ID})...")
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)

    sam_model = sam_processor = None
    if args.variant == "sam3d":
        from transformers import SamModel, SamProcessor

        from config import SAM_MODEL_ID
        from perception.pipeline_sam3d import run_perception_sam3d

        print(f"[EXP1] Variant 'sam3d': Loading SAM ({SAM_MODEL_ID})...")
        sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
        sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)

    n_ok = n_skip = n_err = 0
    for i, session_path in enumerate(sessions, 1):
        scene_id = Path(session_path).name
        out_path = run_dir / "preds" / f"{scene_id}.npz"
        if args.resume and out_path.exists():
            n_skip += 1
            continue
        print(f"[{i}/{len(sessions)}] {scene_id}")
        try:
            if args.variant == "sam3d":
                art = run_perception_sam3d(
                    session_path,
                    dino_model,
                    dino_processor,
                    sam_model,
                    sam_processor,
                )
            else:
                art = run_perception(
                    session_path,
                    dino_model,
                    dino_processor,
                    visualize=args.gui,
                )
            save_artifacts(run_dir, scene_id, art)
            n_ok += 1
        except Exception as exc:
            print(f"[EXP1] ERROR {scene_id}: {exc}")
            n_err += 1

    manifest_out = {
        "data_root": get_data_root(),
        "test_set": args.test_set,
        "variant": args.variant,
        "gui": args.gui,
        "n_sessions": len(sessions),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_error": n_err,
        "device": device,
    }
    with open(run_dir / "inference_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"[EXP1] Done: ok={n_ok} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()
