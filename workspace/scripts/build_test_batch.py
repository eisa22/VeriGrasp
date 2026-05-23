#!/usr/bin/env python3
"""Kleine Testbatch: 2× pallet_rgbd + 2× Box-Is → Data/test_batch_rgbd/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from perception.dataset_unified import build_unified_dataset

# Repräsentative Defaults (Pallet: unterschiedliche Komplexität; Box: mit mehreren Instanzen)
DEFAULT_PALLET = ["Replicator_07", "Replicator_10"]
DEFAULT_BOX = ["230703_145053", "230703_150433"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Testbatch mit 2 Pallet- und 2 Box-Is-Frames bauen"
    )
    parser.add_argument(
        "--out",
        default="Data/test_batch_rgbd",
        help="Zielordner (manifest.json + depths/)",
    )
    parser.add_argument("--pallet", default="Data/pallet_rgbd_data")
    parser.add_argument("--box", default="Data/Box-Is selected")
    parser.add_argument(
        "--pallet-sessions",
        nargs="*",
        default=DEFAULT_PALLET,
        help=f"Pallet-Sessions (default: {' '.join(DEFAULT_PALLET)})",
    )
    parser.add_argument(
        "--box-stems",
        nargs="*",
        default=DEFAULT_BOX,
        help=f"Box-Is Stems ohne Endung (default: {' '.join(DEFAULT_BOX)})",
    )
    parser.add_argument(
        "--skip-depth-gen",
        action="store_true",
        help="PLY→Depth überspringen (nur Manifest)",
    )
    args = parser.parse_args()

    workspace = _ROOT
    unified = workspace / args.out
    pallet = workspace / args.pallet
    box = workspace / args.box

    if not pallet.is_dir():
        raise FileNotFoundError(f"Pallet-Daten nicht gefunden: {pallet}")
    if not box.is_dir():
        raise FileNotFoundError(f"Box-Is nicht gefunden: {box}")

    print("Testbatch:")
    print(f"  Pallet ({len(args.pallet_sessions)}): {', '.join(args.pallet_sessions)}")
    print(f"  Box-Is ({len(args.box_stems)}):     {', '.join(args.box_stems)}")
    print(f"  Output: {unified}\n")

    manifest = build_unified_dataset(
        unified,
        pallet,
        box,
        write_depths=not args.skip_depth_gen,
        pallet_sessions=list(args.pallet_sessions),
        box_stems=list(args.box_stems),
    )

    print(f"Fertig: {manifest['num_frames']} Frames → {unified / 'manifest.json'}")
    for fr in manifest["frames"]:
        print(f"  [{fr['image_id']}] {fr['frame_id']} ({fr['source']})")

    print("\nPipeline:")
    print(
        "  .venv/bin/python3 scripts/run_pipeline.py \\\n"
        f"    --config perception/configs/config_C.yaml \\\n"
        f"    --dataset {args.out} \\\n"
        "    --out runs/test_batch_C/"
    )
    print("\nViewer:")
    print(
        f"  .venv/bin/python3 scripts/view_predictions.py "
        f"--dataset {args.out} --run runs/test_batch_C"
    )


if __name__ == "__main__":
    main()
