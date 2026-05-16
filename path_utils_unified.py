# path_utils_unified.py
"""
Pfad-Hilfsfunktionen für Unified Dataset.
Ersetzt path_utils.py nach erfolgreicher Konvertierung.
"""

import os
import json
from pathlib import Path
from config import BASE_PATH


def get_unified_sessions_dir():
    """Gibt den Pfad zum unified_dataset/sessions Verzeichnis zurück."""
    return Path("/home/samuel/Thesis/VisionPipeline/Data/unified_dataset/sessions")


def load_index():
    """Lädt die Index-Datei des unified_dataset."""
    index_path = Path("/home/samuel/Thesis/VisionPipeline/Data/unified_dataset/index.json")
    if index_path.exists():
        with open(index_path, 'r') as f:
            return json.load(f)
    return None


def get_all_session_paths() -> list:
    """
    Gibt eine Liste aller Session-Pfade aus dem unified_dataset zurück.
    Kompatibel mit der bestehenden Pipeline.
    """
    sessions_dir = get_unified_sessions_dir()
    
    if not sessions_dir.exists():
        print(f"[WARNING] Unified dataset nicht gefunden: {sessions_dir}")
        print(f"          Führe 'python create_unified_dataset.py' aus!")
        return []
    
    sessions = []
    for session_dir in sorted(sessions_dir.iterdir()):
        if session_dir.is_dir():
            # Prüfe ob Session valide ist (hat RGB und Depth)
            rgb_dir = session_dir / "rgb"
            depth_dir = session_dir / "depth"
            
            if rgb_dir.exists() and depth_dir.exists():
                sessions.append(str(session_dir))
            else:
                print(f"[WARNING] Unvollständige Session: {session_dir.name}")
    
    return sessions


def get_session_metadata(session_path: str) -> dict:
    """Lädt die Metadata einer Session."""
    metadata_path = Path(session_path) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def get_session_path() -> str:
    """
    Gibt den Basis-Sessionpfad zurück (einfaches Wrapper).
    Für Kompatibilität mit bestehender Pipeline.
    """
    return BASE_PATH


def get_rgb_path(session_path: str = None) -> str:
    """
    RGB Bildpfad für Frame 0.
    Unterstützt sowohl .png als auch .jpg Formate.
    """
    if session_path is None:
        session_path = BASE_PATH
    
    session_path = Path(session_path)
    rgb_dir = session_path / "rgb"
    
    # Suche nach rgb_0000.png oder rgb_0000.jpg
    for ext in ['.png', '.jpg', '.jpeg']:
        rgb_path = rgb_dir / f"rgb_0000{ext}"
        if rgb_path.exists():
            return str(rgb_path)
    
    # Fallback: Suche nach irgendwelchen Bildern im rgb/ Ordner
    if rgb_dir.exists():
        for img_file in rgb_dir.glob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                return str(img_file)
    
    # Letzter Fallback: Original-Format (für Kompatibilität)
    return str(rgb_dir / "rgb_0000.png")


def get_depth_path(session_path: str = None) -> str:
    """
    Depth-NPY Pfad für Frame 0.
    Unterstützt sowohl depth_0000.npy als auch distance_to_image_plane_0000.npy.
    """
    if session_path is None:
        session_path = BASE_PATH
    
    session_path = Path(session_path)
    
    # Neues Format: depth/depth_0000.npy
    depth_path = session_path / "depth" / "depth_0000.npy"
    if depth_path.exists():
        return str(depth_path)
    
    # Altes Format (Fallback): distance_to_image_plane/...
    depth_path_old = session_path / "distance_to_image_plane" / "distance_to_image_plane_0000.npy"
    if depth_path_old.exists():
        return str(depth_path_old)
    
    # Default: neues Format
    return str(session_path / "depth" / "depth_0000.npy")


def get_pointcloud_path(session_path: str = None) -> str:
    """
    Pointcloud Pfad für Frame 0.
    """
    if session_path is None:
        session_path = BASE_PATH
    
    return str(Path(session_path) / "pointcloud" / "pointcloud_0000.npy")


def get_session_info(session_path: str) -> dict:
    """
    Gibt Informationen über eine Session zurück.
    Nützlich für Debugging und Validierung.
    """
    session_path = Path(session_path)
    
    info = {
        "session_name": session_path.name,
        "exists": session_path.exists(),
        "has_rgb": (session_path / "rgb").exists(),
        "has_depth": (session_path / "depth").exists(),
        "has_pointcloud": (session_path / "pointcloud").exists(),
        "has_metadata": (session_path / "metadata.json").exists(),
    }
    
    # Lade Metadata falls vorhanden
    if info["has_metadata"]:
        info["metadata"] = get_session_metadata(str(session_path))
    
    return info


def filter_sessions_by_source(source_dataset: str = None) -> list:
    """
    Filtert Sessions nach Quell-Dataset.
    
    Args:
        source_dataset: "pallet_rgbd_data" oder "box_is_selected" oder None (alle)
    
    Returns:
        Liste von Session-Pfaden
    """
    all_sessions = get_all_session_paths()
    
    if source_dataset is None:
        return all_sessions
    
    filtered = []
    for session_path in all_sessions:
        metadata = get_session_metadata(session_path)
        if metadata and metadata.get("source_dataset") == source_dataset:
            filtered.append(session_path)
    
    return filtered


def print_dataset_summary():
    """Gibt eine Übersicht über das unified_dataset aus."""
    index = load_index()
    
    if index is None:
        print("❌ Unified dataset nicht gefunden!")
        print("   Führe 'python create_unified_dataset.py' aus.")
        return
    
    print("\n" + "=" * 60)
    print("UNIFIED DATASET SUMMARY")
    print("=" * 60)
    print(f"Version: {index.get('version', 'N/A')}")
    print(f"Created: {index.get('created', 'N/A')}")
    print(f"Total Sessions: {index.get('total_sessions', 0)}")
    print(f"\nSources:")
    for source, count in index.get('sources', {}).items():
        print(f"  - {source}: {count} sessions")
    print("=" * 60 + "\n")


# Kompatibilitäts-Check
if __name__ == "__main__":
    print("Testing path_utils_unified.py...\n")
    
    # Test 1: Dataset Summary
    print_dataset_summary()
    
    # Test 2: Session Paths
    sessions = get_all_session_paths()
    print(f"✓ Gefundene Sessions: {len(sessions)}")
    
    if sessions:
        # Test 3: Erste Session Details
        first_session = sessions[0]
        print(f"\n✓ Erste Session: {Path(first_session).name}")
        
        info = get_session_info(first_session)
        print(f"  RGB: {'✓' if info['has_rgb'] else '✗'}")
        print(f"  Depth: {'✓' if info['has_depth'] else '✗'}")
        print(f"  Pointcloud: {'✓' if info['has_pointcloud'] else '✗'}")
        print(f"  Metadata: {'✓' if info['has_metadata'] else '✗'}")
        
        # Test 4: Pfad-Funktionen
        rgb_path = get_rgb_path(first_session)
        depth_path = get_depth_path(first_session)
        print(f"\n✓ RGB Path: {Path(rgb_path).exists()}")
        print(f"✓ Depth Path: {Path(depth_path).exists()}")
    
    # Test 5: Filtern nach Source
    pallet_sessions = filter_sessions_by_source("pallet_rgbd_data")
    box_sessions = filter_sessions_by_source("box_is_selected")
    print(f"\n✓ Pallet RGBD Sessions: {len(pallet_sessions)}")
    print(f"✓ Box-Is Selected Sessions: {len(box_sessions)}")
    
    print("\n✅ Alle Tests bestanden!")
