#!/usr/bin/env python3
"""
Verbesserte Depth-Map Generierung aus PLY-Dateien.
Behebt das Problem mit negativen Z-Werten und falscher Projektion.
"""

import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
import json


def generate_depth_from_ply_improved(ply_path, rgb_path, fx=437.04, fy=437.04):
    """
    Generiert eine Depth-Map aus PLY mit korrekter Z-Behandlung.
    
    Args:
        ply_path: Pfad zur PLY-Datei
        rgb_path: Pfad zum RGB-Bild (für Dimensionen)
        fx, fy: Kamera Focal Length
    
    Returns:
        depth_map: 2D numpy array mit Tiefenwerten in Metern
    """
    # Lade PLY
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        print("  ❌ PLY enthält keine Punkte!")
        return None
    
    # Lade RGB für Dimensionen
    rgb_img = Image.open(rgb_path)
    width, height = rgb_img.size
    
    # Kamera-Intrinsics
    cx = width / 2
    cy = height / 2
    
    print(f"  📊 PLY: {len(points)} Punkte")
    print(f"  📷 Image: {width}x{height}")
    print(f"  📐 Z-Range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]m")
    
    # KRITISCH: PLY hat negative Z-Werte (Kamera bei origin, Szene in -Z)
    # Wir müssen -Z als Tiefe verwenden!
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Verwende -z als Tiefe (absolute Werte, da Z negativ ist)
    depth_values = -z  # Jetzt sind Tiefen positiv!
    
    # Filtere Punkte hinter der Kamera oder zu weit weg
    valid_mask = (depth_values > 0) & (depth_values < 10.0)  # 0-10m
    
    if valid_mask.sum() == 0:
        print("  ❌ Keine validen Punkte nach Filterung!")
        return None
    
    x = x[valid_mask]
    y = y[valid_mask]
    depth_values = depth_values[valid_mask]
    
    print(f"  ✓ Nach Filterung: {len(depth_values)} valide Punkte")
    print(f"  ✓ Depth-Range: [{depth_values.min():.3f}, {depth_values.max():.3f}]m")
    
    # Initialisiere Depth-Map
    depth_map = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)
    
    # Projiziere jeden 3D-Punkt auf 2D
    for i in range(len(x)):
        # Projektion: 3D -> 2D
        u = int(x[i] * fx / depth_values[i] + cx)
        v = int(y[i] * fy / depth_values[i] + cy)
        
        # Bounds check
        if 0 <= u < width and 0 <= v < height:
            # Z-Buffer: Nimm nächsten Punkt (kleinste Tiefe)
            if counts[v, u] == 0 or depth_values[i] < depth_map[v, u]:
                depth_map[v, u] = depth_values[i]
                counts[v, u] = 1
    
    # Statistik
    filled_pixels = (counts > 0).sum()
    coverage = filled_pixels / (width * height) * 100
    print(f"  📊 Coverage: {coverage:.1f}% ({filled_pixels} Pixel)")
    
    if filled_pixels == 0:
        print("  ❌ Keine Pixel projiziert!")
        return None
    
    # Hole füllen mit Inpainting
    if coverage < 95:
        depth_map = fill_depth_holes_improved(depth_map, counts > 0)
        print(f"  ✓ Löcher gefüllt (Inpainting)")
    
    return depth_map


def fill_depth_holes_improved(depth_map, valid_mask):
    """Füllt Löcher in der Depth-Map mit iterativem Inpainting."""
    from scipy.ndimage import binary_dilation, distance_transform_edt
    
    # Dilate valid mask leicht, um kleine Löcher zu füllen
    dilated_mask = binary_dilation(valid_mask, iterations=2)
    
    # Für größere Löcher: Nearest-Neighbor
    if valid_mask.sum() > 0:
        indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
        filled_depth = depth_map[tuple(indices)]
        
        # Nur Löcher innerhalb der dilated mask füllen
        depth_map = np.where(dilated_mask & ~valid_mask, filled_depth, depth_map)
    
    return depth_map


def regenerate_all_box_selected_depths():
    """Regeneriert alle Depth-Maps für Box-Is selected Sessions."""
    print("=" * 80)
    print("DEPTH-MAP REGENERIERUNG - Box-Is selected")
    print("=" * 80)
    
    unified_sessions = Path("/home/samuel/Thesis/VisionPipeline/Data/unified_dataset/sessions")
    
    # Finde alle box_selected Sessions
    box_sessions = sorted([d for d in unified_sessions.iterdir() 
                          if d.is_dir() and d.name.startswith("box_selected_")])
    
    print(f"\n📦 Gefunden: {len(box_sessions)} Box-Is selected Sessions\n")
    
    success_count = 0
    fail_count = 0
    
    for i, session_dir in enumerate(box_sessions, 1):
        print(f"[{i}/{len(box_sessions)}] {session_dir.name}")
        
        # Lade Metadata für Original-Pfade
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            print("  ⚠️ Keine metadata.json")
            fail_count += 1
            continue
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Finde Original PLY
        original_path = Path(metadata.get("original_path", ""))
        original_files = metadata.get("original_files", {})
        ply_name = original_files.get("ply")
        rgb_name = original_files.get("rgb")
        
        if not ply_name or not rgb_name:
            print("  ⚠️ PLY/RGB Namen nicht in metadata")
            fail_count += 1
            continue
        
        ply_path = original_path / ply_name
        rgb_path = original_path / rgb_name
        
        if not ply_path.exists():
            print(f"  ❌ PLY nicht gefunden: {ply_path}")
            fail_count += 1
            continue
        
        # Generiere Depth-Map
        try:
            depth_map = generate_depth_from_ply_improved(ply_path, rgb_path)
            
            if depth_map is not None and depth_map.max() > 0:
                # Speichere Depth-Map
                depth_dst = session_dir / "depth" / "depth_0000.npy"
                depth_dst.parent.mkdir(exist_ok=True)
                np.save(depth_dst, depth_map)
                
                print(f"  ✅ Depth gespeichert: Range [{depth_map.min():.3f}, {depth_map.max():.3f}]m\n")
                success_count += 1
            else:
                print(f"  ❌ Depth-Generierung fehlgeschlagen\n")
                fail_count += 1
                
        except Exception as e:
            print(f"  ❌ Fehler: {e}\n")
            fail_count += 1
    
    # Zusammenfassung
    print("=" * 80)
    print(f"✅ Erfolgreich: {success_count}/{len(box_sessions)}")
    print(f"❌ Fehlgeschlagen: {fail_count}/{len(box_sessions)}")
    print("=" * 80)


if __name__ == "__main__":
    regenerate_all_box_selected_depths()
