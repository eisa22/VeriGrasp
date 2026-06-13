# path_utils.py
import os
from pathlib import Path

import numpy as np

from config import BASE_PATH


def get_data_root() -> str:
    """Root-Ordner des Blender-Datensatzes (enthält dataset_meta.json + scene_*)."""
    return BASE_PATH


def get_all_session_paths() -> list[str]:
    """Gibt alle Szenenordner scene_* im Blender-Datensatz zurück."""
    data_dir = Path(BASE_PATH).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Datensatzordner nicht gefunden: {data_dir}")

    sessions = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and item.name.startswith("scene_"):
            sessions.append(str(item))
    if not sessions:
        raise FileNotFoundError(f"Keine scene_* Ordner in {data_dir}")
    return sessions


def get_session_path() -> str:
    """Gibt den Basis-Sessionpfad zurück (erste Szene im Datensatz)."""
    sessions = get_all_session_paths()
    return sessions[0]


def get_rgb_path(session_path: str = None) -> str:
    """RGB-Bildpfad (Blender: rgb.png im Szenenordner)."""
    if session_path is None:
        session_path = get_session_path()
    return os.path.join(session_path, "rgb.png")


def get_depth_path(session_path: str = None) -> str:
    """Depth-NPY Pfad (Blender: depth.npy im Szenenordner)."""
    if session_path is None:
        session_path = get_session_path()
    return os.path.join(session_path, "depth.npy")


def get_ground_truth_path(session_path: str = None) -> str:
    if session_path is None:
        session_path = get_session_path()
    return os.path.join(session_path, "ground_truth.json")


def load_session_depth(session_path: str = None) -> np.ndarray:
    """Lädt das Tiefenbild einer Session (Meter, senkrechte z-Tiefe im Kameraframe)."""
    return np.load(get_depth_path(session_path)).astype(np.float32)
