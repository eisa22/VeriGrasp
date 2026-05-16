#!/usr/bin/env python3
"""
Test Script ohne interaktive Visualisierung.
Verarbeitet Session und zeigt nur Konsolen-Output.
"""

import sys
import torch
from transformers import (
    AutoProcessor as DinoProcessor,
    AutoModelForZeroShotObjectDetection as DinoModel
)
from path_utils import get_all_session_paths, filter_sessions_by_source
from config import DINO_MODEL_ID, DEBUG
from main import process_session

# Temporär DEBUG ausschalten
import config
config.DEBUG = False


def test_session_no_viz(session_index=0, source_filter=None):
    """Testet Session ohne Visualisierung."""
    print("=" * 80)
    print("SESSION TEST (Ohne Visualisierung)")
    print("=" * 80)
    
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
        print(f"❌ Session Index {session_index} zu groß")
        return
    
    session_path = sessions[session_index]
    session_name = session_path.split("/")[-1]
    
    print(f"\n📦 Teste: {session_name}")
    
    # DINO Laden
    print(f"\n🤖 Lade DINO...")
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)
    
    # Pipeline
    print(f"\n{'=' * 80}")
    print("PIPELINE")
    print("=" * 80)
    
    result = process_session(session_path, dino_model, dino_processor)
    
    print(f"\n{'=' * 80}")
    print("✅ FERTIG")
    print("=" * 80)


if __name__ == "__main__":
    session_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    source = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_session_no_viz(session_idx, source)
