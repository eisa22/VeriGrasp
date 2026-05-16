#!/usr/bin/env python3
"""
Unified Dataset Converter

Konvertiert beide Datensätze in ein einheitliches Format:
- pallet_rgbd_data (8 Sessions)
- Box-Is selected (51 Sessions)

Ziel: 59 Sessions im unified_dataset/sessions/ Format
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d
from datetime import datetime


class UnifiedDatasetConverter:
    def __init__(self, output_base="Data/unified_dataset"):
        self.output_base = Path(output_base)
        self.sessions_dir = self.output_base / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Kamera-Intrinsics (aus der Pipeline)
        self.default_fx = 437.04
        self.default_fy = 437.04
        
        # Session Registry
        self.session_registry = []
    
    def convert_pallet_rgbd_session(self, replicator_path):
        """Konvertiert eine Replicator Session in unified format."""
        replicator_path = Path(replicator_path)
        session_name = f"pallet_rgbd_{replicator_path.name}"
        
        print(f"\n📦 Konvertiere: {replicator_path.name} -> {session_name}")
        
        # Ziel-Ordner
        session_out = self.sessions_dir / session_name
        session_out.mkdir(exist_ok=True)
        
        # 1. RGB kopieren
        rgb_src = replicator_path / "rgb" / "rgb_0000.png"
        rgb_dst = session_out / "rgb"
        rgb_dst.mkdir(exist_ok=True)
        if rgb_src.exists():
            shutil.copy2(rgb_src, rgb_dst / "rgb_0000.png")
            print(f"  ✓ RGB kopiert")
        else:
            print(f"  ⚠️ RGB nicht gefunden: {rgb_src}")
            return None
        
        # 2. Depth kopieren
        depth_src = replicator_path / "distance_to_image_plane" / "distance_to_image_plane_0000.npy"
        depth_dst = session_out / "depth"
        depth_dst.mkdir(exist_ok=True)
        if depth_src.exists():
            shutil.copy2(depth_src, depth_dst / "depth_0000.npy")
            print(f"  ✓ Depth kopiert")
        else:
            print(f"  ⚠️ Depth nicht gefunden: {depth_src}")
            return None
        
        # 3. Pointcloud kopieren (optional, da wir es aus RGB+Depth generieren können)
        pc_src = replicator_path / "pointcloud" / "pointcloud_0000.npy"
        pc_dst = session_out / "pointcloud"
        pc_dst.mkdir(exist_ok=True)
        if pc_src.exists():
            shutil.copy2(pc_src, pc_dst / "pointcloud_0000.npy")
            
            # Auch normals kopieren falls vorhanden
            normals_src = replicator_path / "pointcloud" / "pointcloud_normals_0000.npy"
            if normals_src.exists():
                shutil.copy2(normals_src, pc_dst / "pointcloud_normals_0000.npy")
            print(f"  ✓ Pointcloud kopiert")
        
        # 4. Metadata erstellen
        rgb_img = Image.open(rgb_src)
        width, height = rgb_img.size
        
        metadata = {
            "session_id": session_name,
            "source_dataset": "pallet_rgbd_data",
            "original_path": str(replicator_path.absolute()),
            "format": {
                "rgb": "png",
                "depth": "npy",
                "pointcloud": "npy",
                "depth_unit": "meters"
            },
            "camera_intrinsics": {
                "fx": self.default_fx,
                "fy": self.default_fy,
                "cx": width / 2,
                "cy": height / 2,
                "width": width,
                "height": height
            },
            "conversion_date": datetime.now().isoformat()
        }
        
        with open(session_out / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Metadata erstellt")
        
        self.session_registry.append(metadata)
        return session_name
    
    def convert_box_is_selected_session(self, rgb_path, ply_path):
        """Konvertiert eine Box-Is selected Session in unified format."""
        rgb_path = Path(rgb_path)
        ply_path = Path(ply_path)
        
        base_name = rgb_path.stem
        session_name = f"box_selected_{base_name}"
        
        print(f"\n📦 Konvertiere: {base_name} -> {session_name}")
        
        # Ziel-Ordner
        session_out = self.sessions_dir / session_name
        session_out.mkdir(exist_ok=True)
        
        # 1. RGB kopieren
        rgb_dst = session_out / "rgb"
        rgb_dst.mkdir(exist_ok=True)
        shutil.copy2(rgb_path, rgb_dst / f"rgb_0000{rgb_path.suffix}")
        print(f"  ✓ RGB kopiert")
        
        # 2. PLY -> Depth + Pointcloud extrahieren
        try:
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            if len(points) == 0:
                print(f"  ❌ PLY enthält keine Punkte!")
                return None
            
            print(f"  📊 PLY: {len(points)} Punkte")
            
            # Pointcloud speichern
            pc_dst = session_out / "pointcloud"
            pc_dst.mkdir(exist_ok=True)
            np.save(pc_dst / "pointcloud_0000.npy", points)
            if len(colors) > 0:
                np.save(pc_dst / "pointcloud_rgb_0000.npy", colors)
            print(f"  ✓ Pointcloud extrahiert")
            
            # Depth-Map aus PLY generieren (Projektion auf 2D)
            depth_map = self._generate_depth_from_pointcloud(points, rgb_path)
            if depth_map is not None:
                depth_dst = session_out / "depth"
                depth_dst.mkdir(exist_ok=True)
                np.save(depth_dst / "depth_0000.npy", depth_map)
                print(f"  ✓ Depth-Map generiert")
            else:
                print(f"  ⚠️ Depth-Map konnte nicht generiert werden")
            
        except Exception as e:
            print(f"  ❌ Fehler beim PLY-Lesen: {e}")
            return None
        
        # 3. Metadata erstellen
        rgb_img = Image.open(rgb_path)
        width, height = rgb_img.size
        
        metadata = {
            "session_id": session_name,
            "source_dataset": "box_is_selected",
            "original_path": str(rgb_path.parent.absolute()),
            "original_files": {
                "rgb": rgb_path.name,
                "ply": ply_path.name
            },
            "format": {
                "rgb": rgb_path.suffix[1:],  # jpg oder png
                "depth": "npy (generated)",
                "pointcloud": "npy (from ply)",
                "depth_unit": "meters"
            },
            "camera_intrinsics": {
                "fx": self.default_fx,
                "fy": self.default_fy,
                "cx": width / 2,
                "cy": height / 2,
                "width": width,
                "height": height,
                "note": "Intrinsics assumed from pipeline defaults"
            },
            "conversion_date": datetime.now().isoformat()
        }
        
        with open(session_out / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Metadata erstellt")
        
        self.session_registry.append(metadata)
        return session_name
    
    def _generate_depth_from_pointcloud(self, points, rgb_path):
        """
        Generiert eine Depth-Map aus einer 3D-Punktwolke.
        Verwendet die Kamera-Intrinsics zur Projektion.
        """
        try:
            # Lade RGB für Dimensionen
            rgb_img = Image.open(rgb_path)
            width, height = rgb_img.size
            
            # Kamera-Intrinsics
            fx = self.default_fx
            fy = self.default_fy
            cx = width / 2
            cy = height / 2
            
            # Initialisiere Depth-Map
            depth_map = np.zeros((height, width), dtype=np.float32)
            counts = np.zeros((height, width), dtype=np.int32)
            
            # Projiziere jeden 3D-Punkt auf 2D
            for point in points:
                x, y, z = point
                
                if z <= 0:  # Punkte hinter der Kamera ignorieren
                    continue
                
                # Projektion: 3D -> 2D
                u = int(x * fx / z + cx)
                v = int(y * fy / z + cy)
                
                # Bounds check
                if 0 <= u < width and 0 <= v < height:
                    # Akkumuliere Tiefe (für mehrere Punkte pro Pixel)
                    depth_map[v, u] += z
                    counts[v, u] += 1
            
            # Mittelwert für Pixel mit mehreren Punkten
            mask = counts > 0
            depth_map[mask] /= counts[mask]
            
            # Hole füllen (einfaches Inpainting)
            if np.sum(mask) > 0:
                depth_map = self._fill_depth_holes(depth_map, mask)
            
            return depth_map
            
        except Exception as e:
            print(f"    ⚠️ Fehler bei Depth-Generierung: {e}")
            return None
    
    def _fill_depth_holes(self, depth_map, valid_mask):
        """Füllt Löcher in der Depth-Map mit Nearest-Neighbor Interpolation."""
        try:
            from scipy.ndimage import distance_transform_edt
            
            # Finde nächsten validen Wert für jedes Loch
            indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
            filled_depth = depth_map[tuple(indices)]
            
            return filled_depth
        except ImportError:
            print("    ⚠️ scipy nicht verfügbar, Depth-Löcher nicht gefüllt")
            return depth_map
    
    def save_index(self):
        """Speichert die Index-Datei mit allen Sessions."""
        index_data = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "total_sessions": len(self.session_registry),
            "sources": {
                "pallet_rgbd_data": len([s for s in self.session_registry if s["source_dataset"] == "pallet_rgbd_data"]),
                "box_is_selected": len([s for s in self.session_registry if s["source_dataset"] == "box_is_selected"])
            },
            "sessions": self.session_registry
        }
        
        index_path = self.output_base / "index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"\n✅ Index gespeichert: {index_path}")
        print(f"   {index_data['total_sessions']} Sessions registriert")
        return index_path


def main():
    """Hauptfunktion: Konvertiert beide Datensätze."""
    print("=" * 80)
    print("UNIFIED DATASET CONVERTER")
    print("=" * 80)
    
    converter = UnifiedDatasetConverter()
    
    # 1. Konvertiere pallet_rgbd_data Sessions
    print("\n" + "=" * 80)
    print("PHASE 1: pallet_rgbd_data")
    print("=" * 80)
    
    pallet_base = Path("/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data")
    replicator_dirs = sorted([d for d in pallet_base.iterdir() if d.is_dir() and d.name.startswith("Replicator_")])
    
    print(f"\nGefunden: {len(replicator_dirs)} Replicator Sessions")
    
    for rep_dir in replicator_dirs:
        converter.convert_pallet_rgbd_session(rep_dir)
    
    # 2. Konvertiere Box-Is selected Sessions
    print("\n" + "=" * 80)
    print("PHASE 2: Box-Is selected")
    print("=" * 80)
    
    box_base = Path("/home/samuel/Thesis/VisionPipeline/Data/Datav2/Data/Box-Is selected")
    
    # Finde alle RGB-PLY Paare
    jpg_files = sorted(box_base.glob("*.jpg"))
    png_files = sorted(box_base.glob("*.png"))
    rgb_files = jpg_files + png_files
    
    print(f"\nGefunden: {len(rgb_files)} RGB-Dateien")
    
    converted = 0
    for rgb_file in rgb_files:
        ply_file = rgb_file.with_suffix('.ply')
        if ply_file.exists():
            result = converter.convert_box_is_selected_session(rgb_file, ply_file)
            if result:
                converted += 1
        else:
            print(f"\n⚠️ Kein PLY-Partner für {rgb_file.name}")
    
    print(f"\n✅ {converted} Box-Is selected Sessions konvertiert")
    
    # 3. Index speichern
    print("\n" + "=" * 80)
    print("PHASE 3: Index erstellen")
    print("=" * 80)
    
    index_path = converter.save_index()
    
    # 4. Zusammenfassung
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print(f"""
✅ Unified Dataset erstellt!

Struktur:
  {converter.output_base}/
  ├── sessions/          ({len(converter.session_registry)} Sessions)
  │   ├── pallet_rgbd_Replicator_XX/
  │   ├── box_selected_YYYYMMDD_HHMMSS/
  │   └── ...
  └── index.json

Nächste Schritte:
  1. path_utils.py anpassen für unified_dataset
  2. Pipeline testen mit neuen Pfaden
  3. Optional: Datenvalidierung durchführen
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    main()
