# Unified Dataset - Anleitung

## 📋 Übersicht

Dieses Dokument beschreibt die **Unified Dataset Struktur** und wie du beide Datensätze (`pallet_rgbd_data` und `Box-Is selected`) in ein einheitliches Format konvertierst.

## 🎯 Ziel

**Vorher:**
- `Data/pallet_rgbd_data/` - 8 Sessions (RGB + Depth + Pointcloud)
- `Data/Datav2/Data/Box-Is selected/` - 51 Sessions (RGB + PLY)

**Nachher:**
- `Data/unified_dataset/sessions/` - 59 Sessions (einheitliches Format)

## 📊 Daten-Analyse

**Box-Is selected:**
- ✅ 51 komplette RGB-PLY Paare
- ⚠️ Keine separaten Depth-Maps (werden aus PLY generiert)
- ❌ Keine Kamera-Intrinsics in PLY (verwendet Pipeline-Defaults)
- ❌ Keine Ground-Truth Annotations

**pallet_rgbd_data:**
- ✅ 8 Sessions mit vollständigen RGB + Depth + Pointcloud
- ✅ Bekannte Kamera-Intrinsics (fx=fy=437.04)

**Gesamt:** 59 Sessions für die Pipeline

## 🏗️ Unified Dataset Struktur

```
Data/unified_dataset/
├── sessions/
│   ├── pallet_rgbd_Replicator_07/
│   │   ├── rgb/
│   │   │   └── rgb_0000.png
│   │   ├── depth/
│   │   │   └── depth_0000.npy
│   │   ├── pointcloud/
│   │   │   ├── pointcloud_0000.npy
│   │   │   └── pointcloud_normals_0000.npy
│   │   └── metadata.json
│   │
│   ├── box_selected_230703_145053_val2017_1/
│   │   ├── rgb/
│   │   │   └── rgb_0000.jpg
│   │   ├── depth/
│   │   │   └── depth_0000.npy (generiert aus PLY)
│   │   ├── pointcloud/
│   │   │   └── pointcloud_0000.npy (konvertiert von PLY)
│   │   └── metadata.json
│   │
│   └── ... (57 weitere Sessions)
│
└── index.json  # Globaler Index aller Sessions
```

## 🚀 Konvertierungs-Workflow

### Schritt 1: Daten analysieren

```bash
python analyze_datasets.py
```

**Output:**
- Zeigt Anzahl Sessions pro Datensatz
- Findet RGB-PLY Paare
- Prüft PLY-Header auf Kamera-Intrinsics
- Sucht nach Annotations
- Speichert `dataset_analysis_result.json`

### Schritt 2: Unified Dataset erstellen

```bash
python create_unified_dataset.py
```

**Was passiert:**
1. **pallet_rgbd_data Sessions:**
   - Kopiert RGB, Depth, Pointcloud
   - Erstellt metadata.json mit Kamera-Intrinsics

2. **Box-Is selected Sessions:**
   - Kopiert RGB
   - Extrahiert Pointcloud aus PLY
   - **Generiert Depth-Map** aus PLY (3D → 2D Projektion)
   - Erstellt metadata.json

3. Erstellt `index.json` mit allen Sessions

**Dauer:** ~2-5 Minuten (je nach PLY-Größe)

**Speicherplatz:** ~2-3 GB (zusätzlich zu Original-Daten)

### Schritt 3: Validierung

```bash
python validate_unified_dataset.py
```

**Prüft:**
- ✓ Vollständigkeit aller Sessions
- ✓ Dateiformate (RGB, Depth, Pointcloud)
- ✓ Metadata-Integrität
- ✓ Depth-Werte Range (Plausibilität)
- ✓ Statistiken nach Quelle

### Schritt 4: Pipeline anpassen

**Option A: Automatischer Wechsel**

```bash
# Backup des alten path_utils.py
cp path_utils.py path_utils_old.py

# Neues path_utils verwenden
cp path_utils_unified.py path_utils.py

# config.py anpassen (optional)
# BASE_PATH auf erste Session setzen
```

**Option B: Manueller Test**

```python
# In main.py oder test script
from path_utils_unified import get_all_session_paths

sessions = get_all_session_paths()
print(f"Sessions: {len(sessions)}")
```

### Schritt 5: Pipeline testen

```bash
# Mit allen Sessions
python main.py

# Oder nur Box-Is selected
python -c "from path_utils_unified import filter_sessions_by_source; print(filter_sessions_by_source('box_is_selected'))"

# Oder nur pallet_rgbd_data
python -c "from path_utils_unified import filter_sessions_by_source; print(filter_sessions_by_source('pallet_rgbd_data'))"
```

## 📄 Metadata Format

Jede Session hat eine `metadata.json`:

```json
{
  "session_id": "pallet_rgbd_Replicator_07",
  "source_dataset": "pallet_rgbd_data",
  "original_path": "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_07",
  "format": {
    "rgb": "png",
    "depth": "npy",
    "pointcloud": "npy",
    "depth_unit": "meters"
  },
  "camera_intrinsics": {
    "fx": 437.04,
    "fy": 437.04,
    "cx": 640.0,
    "cy": 360.0,
    "width": 1280,
    "height": 720
  },
  "conversion_date": "2026-05-16T10:45:00"
}
```

## 🔧 path_utils_unified.py Features

**Neue Funktionen:**

```python
# Alle Sessions (beide Datensätze)
sessions = get_all_session_paths()

# Nur Box-Is selected
box_sessions = filter_sessions_by_source("box_is_selected")

# Nur pallet_rgbd_data
pallet_sessions = filter_sessions_by_source("pallet_rgbd_data")

# Session Info
info = get_session_info(session_path)

# Dataset Summary
print_dataset_summary()
```

**Kompatibilität:**
- ✅ `get_rgb_path()` - funktioniert mit .png und .jpg
- ✅ `get_depth_path()` - funktioniert mit neuem und altem Format
- ✅ `get_all_session_paths()` - API-kompatibel mit alter Version

## ⚙️ Konfiguration

### Kamera-Intrinsics

Die Pipeline verwendet standardmäßig:

```python
fx = fy = 437.04
cx = width / 2
cy = height / 2
```

Falls Box-Is selected andere Intrinsics hat, passe in `create_unified_dataset.py` an:

```python
# In UnifiedDatasetConverter.__init__()
self.default_fx = 437.04  # Anpassen
self.default_fy = 437.04  # Anpassen
```

### Depth-Generierung aus PLY

Die Depth-Map wird durch **3D → 2D Projektion** generiert:

1. Projiziere jeden 3D-Punkt auf 2D Image Plane
2. Akkumuliere Tiefenwerte pro Pixel
3. Fülle Löcher mit Nearest-Neighbor Interpolation

**Limitierungen:**
- Löcher in der Depth-Map möglich (wenn Punkte fehlen)
- Projektion kann ungenau sein ohne exakte Kamera-Pose

## 🐛 Troubleshooting

### "No sessions found"

```bash
# Prüfe ob unified_dataset existiert
ls -la Data/unified_dataset/sessions/

# Falls nicht, führe Konvertierung aus
python create_unified_dataset.py
```

### "Depth-Map all zeros"

Die PLY → Depth Konvertierung kann fehlschlagen wenn:
- PLY in anderem Koordinatensystem ist
- Punkte außerhalb des Image-Bereichs liegen

**Fix:** Manuelle Prüfung der PLY-Daten:

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("path/to/file.ply")
print(f"Points: {len(pcd.points)}")
print(f"Range: {np.asarray(pcd.points).min(axis=0)} to {np.asarray(pcd.points).max(axis=0)}")
```

### "Session incomplete"

```bash
# Validierung zeigt fehlende Dateien
python validate_unified_dataset.py

# Prüfe Original-Daten
ls -la "Data/Datav2/Data/Box-Is selected/"
```

## 📈 Performance

**Konvertierung:**
- pallet_rgbd_data: ~10s (hauptsächlich Kopieren)
- Box-Is selected: ~2-5 min (PLY lesen + Depth generieren)

**Speicherplatz:**
- Original: ~2 GB
- Unified: ~4-5 GB (mit generierten Depth-Maps)

**Pipeline:**
- Keine Performance-Unterschied (gleicher Datenzugriff)
- Vorteil: Einheitliche Schnittstelle

## ✅ Validierungs-Checkliste

Nach Konvertierung:

- [ ] `python validate_unified_dataset.py` läuft ohne Fehler
- [ ] Alle 59 Sessions vorhanden
- [ ] RGB-Bilder öffnen sich korrekt
- [ ] Depth-Maps haben plausible Werte (0-10m)
- [ ] Metadata vollständig
- [ ] Pipeline läuft mit neuem path_utils

## 🔄 Rückgängig machen

Falls Probleme auftreten:

```bash
# Altes path_utils wiederherstellen
cp path_utils_old.py path_utils.py

# Unified dataset löschen (optional)
rm -rf Data/unified_dataset/
```

## 📚 Weiterführende Schritte

1. **Daten-Augmentation:** Erstelle mehr Sessions durch Rotationen/Crops
2. **Ground-Truth Annotations:** Manuell hinzufügen für Box-Is selected
3. **Kamera-Kalibrierung:** Exakte Intrinsics für Box-Is selected messen
4. **Depth-Verbesserung:** Bessere Algorithmen für PLY → Depth

## 👥 Fragen?

Bei Problemen:
1. Prüfe `dataset_analysis_result.json`
2. Führe Validierung aus
3. Schaue in Session metadata.json
4. Prüfe Original-Daten

---

**Erstellt:** 2026-05-16  
**Version:** 1.0
