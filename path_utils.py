# path_utils.py
import os
from config import BASE_PATH


def get_session_path() -> str:
    """Gibt den Basis-Sessionpfad zurück (einfaches Wrapper)."""
    return BASE_PATH


def get_rgb_path() -> str:
    """RGB Bildpfad für Frame 0."""
    return os.path.join(BASE_PATH, "rgb", "rgb_0000.png")


def get_depth_path() -> str:
    """Depth-NPY Pfad für Frame 0."""
    return os.path.join(
        BASE_PATH,
        "distance_to_image_plane",
        "distance_to_image_plane_0000.npy",
    )
