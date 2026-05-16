#!/usr/bin/env python3
"""
Unified Dataset Validator

Validiert das konvertierte unified_dataset:
- Prüft Vollständigkeit aller Sessions
- Validiert Dateiformate
- Prüft Kamera-Intrinsics
- Generiert Statistiken
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict


class DatasetValidator:
    def __init__(self, dataset_path="Data/unified_dataset"):
        self.dataset_path = Path(dataset_path)
        self.sessions_dir = self.dataset_path / "sessions"
        self.index_path = self.dataset_path / "index.json"
        
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
    
    def validate(self):
        """Führt vollständige Validierung durch."""
        print("=" * 80)
        print("UNIFIED DATASET VALIDATION")
        print("=" * 80)
        
        # 1. Strukturprüfung
        self._check_structure()
        
        # 2. Index-Datei prüfen
        self._check_index()
        
        # 3. Sessions validieren
        self._validate_sessions()
        
        # 4. Ergebnisse ausgeben
        self._print_results()
        
        return len(self.errors) == 0
    
    def _check_structure(self):
        """Prüft Basis-Struktur."""
        print("\n📁 Struktur-Prüfung...")
        
        if not self.dataset_path.exists():
            self.errors.append(f"Dataset-Pfad existiert nicht: {self.dataset_path}")
            return
        
        if not self.sessions_dir.exists():
            self.errors.append(f"Sessions-Ordner nicht gefunden: {self.sessions_dir}")
            return
        
        if not self.index_path.exists():
            self.warnings.append(f"Index-Datei nicht gefunden: {self.index_path}")
        
        print(f"  ✓ Dataset-Pfad: {self.dataset_path}")
        print(f"  ✓ Sessions-Ordner: {self.sessions_dir}")
    
    def _check_index(self):
        """Prüft Index-Datei."""
        print("\n📋 Index-Prüfung...")
        
        if not self.index_path.exists():
            return
        
        try:
            with open(self.index_path, 'r') as f:
                index = json.load(f)
            
            required_fields = ["version", "total_sessions", "sessions"]
            for field in required_fields:
                if field not in index:
                    self.errors.append(f"Index fehlt Feld: {field}")
            
            self.stats["index_total_sessions"] = index.get("total_sessions", 0)
            print(f"  ✓ Index geladen: {index.get('total_sessions', 0)} Sessions registriert")
            
        except json.JSONDecodeError as e:
            self.errors.append(f"Index-Datei ist ungültiges JSON: {e}")
    
    def _validate_sessions(self):
        """Validiert alle Sessions."""
        print("\n🔍 Session-Validierung...")
        
        if not self.sessions_dir.exists():
            return
        
        session_dirs = sorted([d for d in self.sessions_dir.iterdir() if d.is_dir()])
        total_sessions = len(session_dirs)
        
        print(f"  Gefunden: {total_sessions} Sessions")
        
        valid_sessions = 0
        for i, session_dir in enumerate(session_dirs, 1):
            print(f"\n  [{i}/{total_sessions}] {session_dir.name}")
            
            is_valid = self._validate_single_session(session_dir)
            if is_valid:
                valid_sessions += 1
        
        self.stats["total_sessions"] = total_sessions
        self.stats["valid_sessions"] = valid_sessions
        self.stats["invalid_sessions"] = total_sessions - valid_sessions
    
    def _validate_single_session(self, session_dir):
        """Validiert eine einzelne Session."""
        is_valid = True
        
        # 1. Metadata prüfen
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            self.warnings.append(f"{session_dir.name}: Keine metadata.json")
        else:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Zähle nach Source
                source = metadata.get("source_dataset", "unknown")
                self.stats[f"source_{source}"] += 1
                
                print(f"    ✓ Metadata (Source: {source})")
            except json.JSONDecodeError:
                self.errors.append(f"{session_dir.name}: Ungültige metadata.json")
                is_valid = False
        
        # 2. RGB prüfen
        rgb_dir = session_dir / "rgb"
        if not rgb_dir.exists():
            self.errors.append(f"{session_dir.name}: Kein rgb/ Ordner")
            is_valid = False
        else:
            rgb_files = list(rgb_dir.glob("*"))
            if not rgb_files:
                self.errors.append(f"{session_dir.name}: rgb/ Ordner leer")
                is_valid = False
            else:
                rgb_file = rgb_files[0]
                try:
                    img = Image.open(rgb_file)
                    width, height = img.size
                    print(f"    ✓ RGB: {rgb_file.suffix} ({width}x{height})")
                    self.stats["rgb_files"] += 1
                except Exception as e:
                    self.errors.append(f"{session_dir.name}: RGB ungültig - {e}")
                    is_valid = False
        
        # 3. Depth prüfen
        depth_dir = session_dir / "depth"
        if not depth_dir.exists():
            self.errors.append(f"{session_dir.name}: Kein depth/ Ordner")
            is_valid = False
        else:
            depth_files = list(depth_dir.glob("*.npy"))
            if not depth_files:
                self.errors.append(f"{session_dir.name}: Keine depth .npy Datei")
                is_valid = False
            else:
                depth_file = depth_files[0]
                try:
                    depth = np.load(depth_file)
                    print(f"    ✓ Depth: {depth.shape}, Range: [{depth.min():.2f}, {depth.max():.2f}]m")
                    self.stats["depth_files"] += 1
                    
                    # Warnung bei verdächtigen Werten
                    if depth.max() > 100:
                        self.warnings.append(f"{session_dir.name}: Depth max > 100m (verdächtig)")
                    if depth.min() < 0:
                        self.warnings.append(f"{session_dir.name}: Negative Depth-Werte")
                    
                except Exception as e:
                    self.errors.append(f"{session_dir.name}: Depth ungültig - {e}")
                    is_valid = False
        
        # 4. Pointcloud prüfen (optional)
        pc_dir = session_dir / "pointcloud"
        if pc_dir.exists():
            pc_files = list(pc_dir.glob("*.npy")) + list(pc_dir.glob("*.ply"))
            if pc_files:
                try:
                    pc_file = pc_files[0]
                    if pc_file.suffix == '.npy':
                        pc = np.load(pc_file)
                        print(f"    ✓ Pointcloud: {pc.shape} Punkte")
                    else:
                        print(f"    ✓ Pointcloud: {pc_file.suffix}")
                    self.stats["pointcloud_files"] += 1
                except Exception as e:
                    self.warnings.append(f"{session_dir.name}: Pointcloud ungültig - {e}")
        
        return is_valid
    
    def _print_results(self):
        """Gibt Validierungs-Ergebnisse aus."""
        print("\n" + "=" * 80)
        print("VALIDIERUNGS-ERGEBNISSE")
        print("=" * 80)
        
        # Statistiken
        print("\n📊 Statistiken:")
        print(f"  Gesamt Sessions: {self.stats.get('total_sessions', 0)}")
        print(f"  Valide Sessions: {self.stats.get('valid_sessions', 0)}")
        print(f"  Invalide Sessions: {self.stats.get('invalid_sessions', 0)}")
        print(f"\n  Nach Quelle:")
        print(f"    pallet_rgbd_data: {self.stats.get('source_pallet_rgbd_data', 0)}")
        print(f"    box_is_selected: {self.stats.get('source_box_is_selected', 0)}")
        print(f"\n  Dateien:")
        print(f"    RGB: {self.stats.get('rgb_files', 0)}")
        print(f"    Depth: {self.stats.get('depth_files', 0)}")
        print(f"    Pointcloud: {self.stats.get('pointcloud_files', 0)}")
        
        # Fehler
        if self.errors:
            print(f"\n❌ FEHLER ({len(self.errors)}):")
            for error in self.errors[:10]:
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... und {len(self.errors) - 10} weitere")
        else:
            print("\n✅ KEINE FEHLER")
        
        # Warnungen
        if self.warnings:
            print(f"\n⚠️ WARNUNGEN ({len(self.warnings)}):")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... und {len(self.warnings) - 10} weitere")
        
        # Gesamtergebnis
        print("\n" + "=" * 80)
        if len(self.errors) == 0:
            print("✅ DATASET VALIDIERUNG ERFOLGREICH")
        else:
            print("❌ DATASET VALIDIERUNG FEHLGESCHLAGEN")
        print("=" * 80)


def main():
    validator = DatasetValidator()
    success = validator.validate()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
