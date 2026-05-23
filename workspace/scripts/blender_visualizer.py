#!/usr/bin/env python3
"""Launch Blender RGB-D visualizer from the thesis workspace."""

import sys
from pathlib import Path

_BLENDER_VIS_DIR = Path(__file__).resolve().parents[3] / "blender_visualisation"
if str(_BLENDER_VIS_DIR) not in sys.path:
  sys.path.insert(0, str(_BLENDER_VIS_DIR))

from blender_visualizer import BlenderVisualizer, main, resolve_blender_data_dir  # noqa: E402

if __name__ == "__main__":
  main()
