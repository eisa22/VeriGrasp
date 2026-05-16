#!/usr/bin/env python3
"""
Verlinkt Original-PLY-Dateien im unified_dataset.
Statt generierte Depth-Maps zu verwenden, nutzt die Pipeline die Original-PLYs direkt.
"""

import os
import shutil
from pathlib import Path
import json


def link_original_ply_files():
    """Kopiert/verlinkt Original-PLY-Dateien ins unified_dataset."""
    print("=" * 80)
    print("PLY-VERLINKUNG - Box-Is selected Sessions")
    print("=" * 80)
    
    unified_sessions = Path("/home/samuel/Thesis/VisionPipeline/Data/unified_dataset/sessions")
    
    # Finde alle box_selected Sessions
    box_sessions = sorted([d for d in unified_sessions.iterdir() 
                          if d.is_dir() and d.name.startswith("box_selected_")])
    
    print(f"\n📦 Verarbeite: {len(box_sessions)} Box-Is selected Sessions\n")
    
    success_count = 0
    
    for i, session_dir in enumerate(box_sessions, 1):
        print(f"[{i}/{len(box_sessions)}] {session_dir.name}")
        
        # Lade Metadata
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            print("  ⚠️ Keine metadata.json")
            continue
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Finde Original-PLY
        original_path = Path(metadata.get("original_path", ""))
        original_files = metadata.get("original_files", {})
        ply_name = original_files.get("ply")
        
        if not ply_name:
            print("  ⚠️ PLY Name nicht in metadata")
            continue
        
        ply_src = original_path / ply_name
        
        if not ply_src.exists():
            print(f"  ❌ PLY nicht gefunden: {ply_src}")
            continue
        
        # Ziel: pointcloud/ Ordner
        pc_dir = session_dir / "pointcloud"
        pc_dir.mkdir(exist_ok=True)
        
        ply_dst = pc_dir / ply_name
        
        # Symlink oder Kopie erstellen
        if ply_dst.exists():
            if ply_dst.is_symlink() or ply_dst.stat().st_size == ply_src.stat().st_size:
                print(f"  ✓ PLY bereits vorhanden")
                success_count += 1
                continue
        
        try:
            # Symlink erstellen (spart Speicherplatz)
            if ply_dst.exists():
                ply_dst.unlink()
            
            os.symlink(ply_src, ply_dst)
            print(f"  ✅ PLY verlinkt (Symlink)")
            success_count += 1
            
        except OSError:
            # Fallback: Kopieren (falls Symlinks nicht unterstützt)
            shutil.copy2(ply_src, ply_dst)
            print(f"  ✅ PLY kopiert")
            success_count += 1
    
    print(f"\n{'=' * 80}")
    print(f"✅ {success_count}/{len(box_sessions)} Sessions mit PLY verlinkt")
    print("=" * 80)


if __name__ == "__main__":
    link_original_ply_files()
