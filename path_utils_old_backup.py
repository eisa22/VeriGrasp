# path_utils.py
import os
from config import BASE_PATH


def get_all_session_paths() -> list:
    """Gibt eine Liste aller Session-Pfade zurück."""
    data_dir = "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data"
    sessions = []
    
    # Alle Replicator_XX Ordner finden
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith("Replicator_"):
            sessions.append(item_path)
    
    return sessions


def get_session_path() -> str:
    """Gibt den Basis-Sessionpfad zurück (einfaches Wrapper)."""
    return BASE_PATH


def get_rgb_path(session_path: str = None) -> str:
    """RGB Bildpfad für Frame 0."""
    if session_path is None:
        session_path = BASE_PATH
    return os.path.join(session_path, "rgb", "rgb_0000.png")


def get_depth_path(session_path: str = None) -> str:
    """Depth-NPY Pfad für Frame 0."""
    if session_path is None:
        session_path = BASE_PATH
    return os.path.join(
        session_path,
        "distance_to_image_plane",
        "distance_to_image_plane_0000.npy",
    )
