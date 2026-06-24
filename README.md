# Vision-to-Grasp Pipeline

Eine vollständige Pipeline von Zero-shot Objekterkennung bis zu validierten Greifpunkten für industrielle Roboter.

## 🎯 Überblick

Diese Pipeline implementiert ein modulares System für roboterassistiertes Greifen von Paketen:

1. **Perception Layer**: Zero-shot Objekterkennung mit Grounding DINO + SAM
2. **3D Reconstruction**: Konvertierung von 2D-Masken zu 3D-Punktwolken
3. **Grasp Generation**: Generierung von Sauggreif-Kandidaten (SuctionNet)
4. **Verification Layer**: Regelbasierte Validierung durch geometrische Constraints
5. **Visualization**: 3D-Visualisierung der Ergebnisse

## 📁 Projektstruktur

```
VisionPipeline/
├── config.py                      # Zentrale Konfiguration
├── main.py                        # Haupt-Pipeline
├── path_utils.py                  # Pfad-Hilfsfunktionen
│
├── GroundingSAM/                  # Zero-shot Detection
│   └── grounding_sam.py
│
├── Sam3D/                         # 3D Reconstruction
│   └── sam3d.py
│
├── GraspGeneration/               # Grasp Candidate Generation
│   ├── __init__.py
│   └── suction_net.py
│
├── Verification/                  # Geometric Verifier
│   ├── __init__.py
│   └── geometric_verifier.py
│
├── Visualization/                 # 3D Visualisierung
│   ├── __init__.py
│   └── grasp_visualizer.py
│
└── tests/                         # Unit & Integration Tests
    ├── __init__.py
    ├── test_verification.py
    ├── test_integration.py
    └── run_all_tests.py
```

## 🚀 Installation

### Voraussetzungen

- Python 3.9+ (getestet mit 3.9)
- CUDA-fähige GPU optional (CPU funktioniert)
- Virtual Environment: `.venv` im Repo-Root (lokal, nicht im Git)

### Dependencies installieren

```bash
# Virtual Environment aktivieren (vom Repo-Root)
source .venv/bin/activate

# Falls Pakete fehlen:
pip install torch torchvision transformers
pip install open3d numpy pillow opencv-python scikit-learn scipy
```

## 💻 Verwendung

### Haupt-Pipeline ausführen

```bash
source .venv/bin/activate
python main.py
```

**Datensatz:** `Data/blender_dataset/scene_*` (konfiguriert in `config.py` → `BASE_PATH`)

**Batch ohne Visualisierung:**

```bash
python main.py --test      # alle Szenen → Results/
python main.py --test 40   # erste 40 Szenen
```

### Konfiguration anpassen

Bearbeite `config.py` um Parameter anzupassen:

```python
# Perception
TEXT_PROMPT = ["cardboard box", "parcel"]
BOX_THRESHOLD = 0.25

# Grasp Generation
SUCTIONNET_SCORE_THRESHOLD = 0.5
MAX_GRASP_CANDIDATES_PER_OBJECT = 10

# Verification
PLANARITY_THRESHOLD = 0.02      # meters
NORMAL_ANGLE_THRESHOLD = 15.0   # degrees
EDGE_DISTANCE_THRESHOLD = 0.02  # meters

# Output
TOP_N_GRASPS = 5
DEBUG = True  # Aktiviert Visualisierung
```

### Tests ausführen

```bash
# Alle Tests
python tests/run_all_tests.py

# Mit höherem Verbosity
python tests/run_all_tests.py --verbose

# Nur Verification Tests
python -m unittest tests.test_verification

# Nur Integration Tests
python -m unittest tests.test_integration
```

## 🔍 Module im Detail

### 1. Perception Layer (`GroundingSAM/`)

Verwendet Grounding DINO für Zero-shot Objekterkennung und SAM für präzise Segmentierung.

**Input**: RGB-Bild  
**Output**: 2D-Masken, Bounding Boxes, Labels, Scores

### 2. 3D Reconstruction (`Sam3D/`)

Konvertiert 2D-Masken zu 3D-Punktwolken mittels RGB-D-Daten.

**Input**: 2D-Masken, RGB-Bild, Tiefenkarte  
**Output**: 3D-Punktwolken pro Objekt

### 3. Grasp Generation (`GraspGeneration/`)

Generiert Greifkandidaten für Sauggreifer.

**Features**:
- Pre-trained SuctionNet Support (TODO)
- Geometriebasierte Heuristik (aktuell)
- Normalen-Schätzung
- Score-basiertes Ranking

**Input**: 3D-Punktwolke  
**Output**: Liste von GraspCandidate Objekten

### 4. Verification Layer (`Verification/`)

Regelbasierte Validierung durch geometrische Constraints.

**Checks**:
- ✓ Planarity (lokale Oberflächenkrümmung)
- ✓ Surface Normal Alignment
- ✓ Edge Proximity (Occlusion Risk)
- ✓ Sensor Consistency
- ✓ Workspace Bounds

**Input**: Greifkandidaten, Punktwolken, Normalen, Depth  
**Output**: Validierte Grasps mit Rejection Reasons

### 5. Visualization (`Visualization/`)

3D-Visualisierung mit Open3D.

**Features**:
- Farbcodierung (Grün=Valid, Rot=Rejected)
- Greifnormalen als Pfeile
- Heatmap-Modus
- Multi-Object Support

## 📊 Datenformat

Die Pipeline erwartet folgende Datenstruktur:

```
Data/pallet_rgbd_data/
└── Replicator_XX/
    ├── rgb/
    │   └── rgb_0000.png
    ├── distance_to_image_plane/
    │   └── distance_to_image_plane_0000.npy
    └── pointcloud/
        ├── pointcloud_0000.npy
        ├── pointcloud_normals_0000.npy
        └── ...
```

## 🎨 Visualisierung

Bei `DEBUG = True` in `config.py`:

1. **2D-Visualisierung**: Grounding DINO + SAM Ergebnisse
2. **3D-Visualisierung**: Segmentierte Punktwolken
3. **Grasp-Visualisierung**: Top validierte Greifpunkte mit Normalen

**Steuerung**:
- Maus: Kamera drehen
- Scroll: Zoom
- `Q`: Schließen

## 🔧 Anpassung

### Eigenes SuctionNet-Modell integrieren

```python
# In config.py
SUCTIONNET_MODEL_PATH = "/path/to/your/model.pth"

# In GraspGeneration/suction_net.py
def _load_model(self, model_path):
    self.model = torch.load(model_path)
    self.model.eval()
    # ...
```

### Workspace Bounds definieren

```python
# In config.py
import numpy as np

WORKSPACE_BOUNDS = (
    np.array([-0.5, -0.5, 0.0]),  # min (x, y, z)
    np.array([0.5, 0.5, 1.0])     # max (x, y, z)
)
```

### Verification Constraints anpassen

Alle Schwellwerte können in `config.py` angepasst werden. Für strengere Validierung:

```python
PLANARITY_THRESHOLD = 0.01      # Strikter
NORMAL_ANGLE_THRESHOLD = 10.0   # Enger
EDGE_DISTANCE_THRESHOLD = 0.03  # Größerer Sicherheitsabstand
```

## 📈 Performance

Typische Laufzeiten (auf GPU):

- Grounding DINO + SAM: ~2-3s
- 3D Reconstruction: ~0.5s
- Grasp Generation: ~0.1s pro Objekt
- Verification: ~0.05s pro Kandidat

**Gesamt**: ~3-5s für vollständige Pipeline

## 🐛 Troubleshooting

### "RuntimeError: CUDA out of memory"

Reduziere die Anzahl der Kandidaten:
```python
MAX_GRASP_CANDIDATES_PER_OBJECT = 5
```

### "No grasps passed verification"

Lockere die Constraints:
```python
PLANARITY_THRESHOLD = 0.03
NORMAL_ANGLE_THRESHOLD = 20.0
```

### Visualisierung öffnet nicht

Prüfe ob Display verfügbar ist:
```python
DEBUG = False  # Deaktiviert Visualisierung
```

## 📚 Referenzen

- **Grounding DINO**: [Liu et al. 2023](https://arxiv.org/abs/2303.05499)
- **SAM**: [Kirillov et al. 2023](https://arxiv.org/abs/2304.02643)
- **GraspNet**: [Fang et al. 2020](https://arxiv.org/abs/2003.00226)
- **Open3D**: [Zhou et al. 2018](http://www.open3d.org/)

## 📝 TODO / Erweiterungen

- [ ] Pre-trained SuctionNet-Modell integrieren
- [ ] Robot Execution Interface
- [ ] Batch-Processing für mehrere Frames
- [ ] Collision Detection mit Roboter-Modell
- [ ] ROS Integration
- [ ] Real-time Performance Optimierung

## 👥 Autor

Samuel - Master Thesis

## 📄 Lizenz

Für Forschungszwecke.

