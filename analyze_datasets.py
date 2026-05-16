#!/usr/bin/env python3
"""
Script zur Analyse beider Datensätze.
Findet heraus: Struktur, Anzahl Sessions, Kamera-Intrinsics, Ground-Truth
"""
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import json


def analyze_box_is_selected():
    """Analysiert den Box-Is selected Datensatz."""
    print("=" * 80)
    print("ANALYSE: Box-Is selected Dataset")
    print("=" * 80)
    
    base_path = Path("/home/samuel/Thesis/VisionPipeline/Data/Datav2/Data/Box-Is selected")
    
    if not base_path.exists():
        print(f"❌ Pfad existiert nicht: {base_path}")
        return None
    
    # Alle Dateien sammeln
    all_files = list(base_path.glob("*"))
    
    print(f"\n📊 Gesamt: {len(all_files)} Dateien")
    
    # Nach Dateityp gruppieren
    file_types = defaultdict(list)
    for f in all_files:
        if f.is_file():
            ext = f.suffix.lower()
            file_types[ext].append(f.name)
    
    print("\n📁 Dateitypen:")
    for ext, files in sorted(file_types.items()):
        print(f"  {ext}: {len(files)} Dateien")
        if len(files) <= 5:
            for fname in sorted(files):
                print(f"    - {fname}")
        else:
            for fname in sorted(files)[:3]:
                print(f"    - {fname}")
            print(f"    ... und {len(files) - 3} weitere")
    
    # Paare finden (gleicher Basisname)
    print("\n🔗 RGB-PLY Paare:")
    base_names = {}
    for f in all_files:
        if f.is_file():
            base_name = f.stem  # Name ohne Extension
            if base_name not in base_names:
                base_names[base_name] = []
            base_names[base_name].append(f.suffix)
    
    complete_pairs = []
    for base_name, extensions in base_names.items():
        if '.jpg' in extensions or '.png' in extensions:
            has_rgb = True
        else:
            has_rgb = False
        has_ply = '.ply' in extensions
        
        if has_rgb and has_ply:
            complete_pairs.append(base_name)
    
    print(f"  Komplette Paare (RGB + PLY): {len(complete_pairs)}")
    if complete_pairs:
        for pair in complete_pairs[:5]:
            print(f"    ✓ {pair}")
        if len(complete_pairs) > 5:
            print(f"    ... und {len(complete_pairs) - 5} weitere")
    
    # Versuche PLY-Datei zu analysieren
    ply_files = [f for f in all_files if f.suffix.lower() == '.ply']
    if ply_files:
        print(f"\n🔍 Analysiere erste PLY-Datei: {ply_files[0].name}")
        analyze_ply_file(ply_files[0])
    
    # Suche nach Annotations
    print("\n🏷️ Suche nach Annotations:")
    annotation_files = []
    for pattern in ['*.json', '*.xml', '*.txt', '*.yaml']:
        annotation_files.extend(base_path.glob(pattern))
    
    if annotation_files:
        print(f"  Gefunden: {len(annotation_files)} Annotations-Dateien")
        for ann_file in annotation_files[:5]:
            print(f"    - {ann_file.name}")
    else:
        print("  ❌ Keine offensichtlichen Annotations-Dateien gefunden")
    
    return {
        "path": str(base_path),
        "total_files": len(all_files),
        "file_types": {k: len(v) for k, v in file_types.items()},
        "complete_pairs": len(complete_pairs),
        "pair_names": complete_pairs,
        "annotation_files": [f.name for f in annotation_files]
    }


def analyze_ply_file(ply_path):
    """Analysiert eine PLY-Datei im Detail."""
    try:
        with open(ply_path, 'rb') as f:
            # Lese Header
            header_lines = []
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
                if len(header_lines) > 100:  # Safety
                    break
        
        print("  PLY Header:")
        # Zeige wichtige Header-Zeilen
        for line in header_lines[:20]:
            if any(keyword in line.lower() for keyword in ['format', 'element', 'property', 'comment', 'camera']):
                print(f"    {line}")
        
        if len(header_lines) > 20:
            print(f"    ... ({len(header_lines)} Zeilen gesamt)")
        
        # Versuche Kamera-Intrinsics zu finden
        camera_info = []
        for line in header_lines:
            if 'camera' in line.lower() or 'intrinsic' in line.lower() or 'focal' in line.lower():
                camera_info.append(line)
        
        if camera_info:
            print("\n  📷 Kamera-Informationen gefunden:")
            for info in camera_info:
                print(f"    {info}")
        else:
            print("\n  ⚠️ Keine Kamera-Intrinsics im PLY-Header gefunden")
        
        # Versuche Punkt-Anzahl zu extrahieren
        for line in header_lines:
            if 'element vertex' in line.lower():
                parts = line.split()
                if len(parts) >= 3:
                    num_vertices = parts[2]
                    print(f"\n  📈 Anzahl Vertices: {num_vertices}")
        
        # Eigenschaften der Vertices
        properties = [line for line in header_lines if line.startswith('property')]
        if properties:
            print(f"\n  🔧 Vertex Properties ({len(properties)}):")
            for prop in properties[:10]:
                print(f"    {prop}")
            if len(properties) > 10:
                print(f"    ... und {len(properties) - 10} weitere")
                
    except Exception as e:
        print(f"  ❌ Fehler beim Analysieren: {e}")


def analyze_pallet_rgbd():
    """Analysiert den pallet_rgbd_data Datensatz."""
    print("\n" + "=" * 80)
    print("ANALYSE: pallet_rgbd_data Dataset")
    print("=" * 80)
    
    base_path = Path("/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data")
    
    if not base_path.exists():
        print(f"❌ Pfad existiert nicht: {base_path}")
        return None
    
    # Finde alle Replicator Ordner
    replicator_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("Replicator_")])
    
    print(f"\n📊 Gefunden: {len(replicator_dirs)} Replicator Sessions")
    
    if replicator_dirs:
        # Analysiere erste Session im Detail
        sample_dir = replicator_dirs[0]
        print(f"\n🔍 Detailanalyse: {sample_dir.name}")
        
        subdirs = [d for d in sample_dir.iterdir() if d.is_dir()]
        print(f"  Unterordner:")
        for subdir in subdirs:
            files = list(subdir.glob("*"))
            print(f"    📁 {subdir.name}/: {len(files)} Dateien")
            
            # Zeige Beispiel-Dateien
            if files:
                for f in files[:3]:
                    print(f"      - {f.name}")
                if len(files) > 3:
                    print(f"      ... und {len(files) - 3} weitere")
        
        # Suche nach metadata oder camera info
        print(f"\n🔍 Suche nach Metadata:")
        for pattern in ['*.json', '*.txt', '*.yaml', '*.xml']:
            metadata_files = list(sample_dir.glob(pattern))
            if metadata_files:
                for mf in metadata_files:
                    print(f"  📄 {mf.name}")
                    if mf.suffix == '.json':
                        try:
                            with open(mf, 'r') as f:
                                data = json.load(f)
                                print(f"    Content: {json.dumps(data, indent=2)[:200]}...")
                        except:
                            pass
    
    return {
        "path": str(base_path),
        "num_sessions": len(replicator_dirs),
        "session_names": [d.name for d in replicator_dirs]
    }


def compare_datasets(box_data, pallet_data):
    """Vergleicht beide Datensätze."""
    print("\n" + "=" * 80)
    print("VERGLEICH & EMPFEHLUNG")
    print("=" * 80)
    
    print("\n📊 Zusammenfassung:")
    print(f"  Box-Is selected: {box_data.get('complete_pairs', 0)} Sessions (RGB + PLY)")
    print(f"  pallet_rgbd_data: {pallet_data.get('num_sessions', 0)} Sessions (RGB + Depth + Pointcloud)")
    print(f"  GESAMT: {box_data.get('complete_pairs', 0) + pallet_data.get('num_sessions', 0)} Sessions")
    
    print("\n🎯 Unified Dataset Struktur:")
    print("""
  Data/unified_dataset/
  ├── sessions/
  │   ├── pallet_rgbd_Replicator_01/
  │   │   ├── rgb/
  │   │   │   └── rgb_0000.png
  │   │   ├── depth/
  │   │   │   └── depth_0000.npy
  │   │   ├── pointcloud/
  │   │   │   └── pointcloud_0000.npy
  │   │   └── metadata.json
  │   │
  │   ├── box_selected_230703_145053_val2017_1/
  │   │   ├── rgb/
  │   │   │   └── rgb_0000.jpg
  │   │   ├── depth/
  │   │   │   └── depth_0000.npy (aus PLY extrahiert)
  │   │   ├── pointcloud/
  │   │   │   └── pointcloud_0000.ply (original)
  │   │   └── metadata.json
  │   │
  │   └── ...
  │
  └── index.json  # Liste aller Sessions mit Metadata
    """)


def main():
    """Hauptfunktion."""
    print("\n🔬 DATENSATZ ANALYSE TOOL")
    print("Analysiere beide Datensätze für Unified Dataset Konzept\n")
    
    # Analysiere beide Datensätze
    box_data = analyze_box_is_selected()
    pallet_data = analyze_pallet_rgbd()
    
    # Vergleich und Empfehlung
    if box_data and pallet_data:
        compare_datasets(box_data, pallet_data)
        
        # Speichere Ergebnis
        result = {
            "box_is_selected": box_data,
            "pallet_rgbd_data": pallet_data,
            "timestamp": "2026-05-16"
        }
        
        output_file = "dataset_analysis_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Analyse gespeichert in: {output_file}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
