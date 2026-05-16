#!/usr/bin/env python3
"""
Test Script: Verarbeitet nur eine Session aus dem unified_dataset.
Nützlich für schnelles Testen ohne alle 59 Sessions zu verarbeiten.
"""

import sys
import torch
from transformers import (
    AutoProcessor as DinoProcessor,
    AutoModelForZeroShotObjectDetection as DinoModel
)
from path_utils import get_all_session_paths, get_session_metadata, filter_sessions_by_source
from config import DINO_MODEL_ID
from main import process_session


def test_single_session(session_index=0, source_filter=None):
    """
    Testet die Pipeline mit einer einzelnen Session.
    
    Args:
        session_index: Index der Session (0-basiert)
        source_filter: Filter nach Quelle ("pallet_rgbd_data" oder "box_is_selected" oder None)
    """
    print("=" * 80)
    print("SINGLE SESSION TEST")
    print("=" * 80)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Sessions laden
    if source_filter:
        sessions = filter_sessions_by_source(source_filter)
        print(f"\nFilter: {source_filter}")
    else:
        sessions = get_all_session_paths()
    
    print(f"Verfügbare Sessions: {len(sessions)}")
    
    if session_index >= len(sessions):
        print(f"❌ Session Index {session_index} zu groß (max: {len(sessions) - 1})")
        return
    
    # Wähle Session
    session_path = sessions[session_index]
    session_name = session_path.split("/")[-1]
    
    print(f"\n📦 Teste Session [{session_index}]: {session_name}")
    
    # Metadata anzeigen
    metadata = get_session_metadata(session_path)
    if metadata:
        print(f"  Quelle: {metadata.get('source_dataset', 'unknown')}")
        print(f"  Format: RGB={metadata.get('format', {}).get('rgb', '?')}, "
              f"Depth={metadata.get('format', {}).get('depth', '?')}")
        intrinsics = metadata.get('camera_intrinsics', {})
        print(f"  Kamera: {intrinsics.get('width', '?')}x{intrinsics.get('height', '?')}, "
              f"fx={intrinsics.get('fx', '?'):.2f}")
    
    # DINO Laden
    print(f"\n🤖 Lade DINO Modell...")
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)
    
    # Pipeline ausführen
    print(f"\n{'=' * 80}")
    print("PIPELINE STARTEN")
    print("=" * 80)
    
    result = process_session(session_path, dino_model, dino_processor)
    
    print(f"\n{'=' * 80}")
    print("✅ TEST ABGESCHLOSSEN")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    # Argumente: session_index und optionaler source_filter
    
    if len(sys.argv) > 1:
        try:
            session_index = int(sys.argv[1])
        except ValueError:
            print("❌ Ungültiger Session Index (muss Zahl sein)")
            sys.exit(1)
    else:
        session_index = 0  # Default: erste Session
    
    source_filter = None
    if len(sys.argv) > 2:
        source_filter = sys.argv[2]
        if source_filter not in ["pallet_rgbd_data", "box_is_selected"]:
            print(f"⚠️ Unbekannter Source Filter: {source_filter}")
            print("   Gültige Werte: pallet_rgbd_data, box_is_selected")
    
    # Test ausführen
    test_single_session(session_index, source_filter)
    
    # Beispiele ausgeben
    print("\n📚 Beispiele:")
    print("  python test_single_session.py                    # Teste erste Session")
    print("  python test_single_session.py 5                  # Teste Session 5")
    print("  python test_single_session.py 0 box_is_selected  # Teste erste Box-Is selected Session")
    print("  python test_single_session.py 2 pallet_rgbd_data # Teste dritte pallet_rgbd Session")
