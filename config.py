# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Blender RGB-D Datensatz (scene_* Unterordner)
BASE_PATH = str(PROJECT_ROOT / "Data" / "blender_dataset")

# Session-Pfad: erste Szene im Datensatz (wird von path_utils aufgelöst)
SESSION_PATH = BASE_PATH

# Kamera-Intrinsics aus dataset_meta.json (synthetic_depal / blender_dataset)
try:
    from camera_intrinsics import load_camera_intrinsics

    _CAM = load_camera_intrinsics(data_root=BASE_PATH)
    CAMERA_FX = float(_CAM["fx"])
    CAMERA_FY = float(_CAM["fy"])
    CAMERA_CX = float(_CAM["cx"])
    CAMERA_CY = float(_CAM["cy"])
    CAMERA_WIDTH = int(_CAM["width"])
    CAMERA_HEIGHT = int(_CAM["height"])
except (FileNotFoundError, KeyError, TypeError):
    CAMERA_FX = CAMERA_FY = 497.77777777777777
    CAMERA_CX = 320.0
    CAMERA_CY = 240.0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

# Debug-Modus für Visualisierung
DEBUG = True

_LOCAL_DINO = PROJECT_ROOT / "LocalModels" / "GroundingDINO"
DINO_MODEL_ID = str(_LOCAL_DINO) if _LOCAL_DINO.is_dir() else "IDEA-Research/grounding-dino-base"
SAM_MODEL_ID  = "facebook/sam-vit-base"

TEXT_PROMPT = [
    "cardboard box",
    "small cardboard box",
    "shipping box",
    "parcel"
]

# DINO: Moderate Thresholds (für Regions, nicht finale Boxen)
BOX_THRESHOLD  = 0.30   # niedriger, da wir nachfiltern
TEXT_THRESHOLD = 0.25

# SAM Grid-Prompts
SAM_GRID_SIZE = 16              # 16x16 Grid pro ROI
SAM_MASK_MIN_AREA = 500         # Mindestgröße in Pixeln
SAM_DEDUPLICATION_IOU = 0.85    # IoU für Duplikat-Erkennung

# Box-Filterung
MAX_BOX_AREA_RATIO = 0.12       # Max 12% der Bildfläche
RELATIVE_IOU_NMS_THRESH = 0.3   # NMS mit relativer IoU

# Tiefenfilterung: PARAMETERFREI
# Die Tiefentrennung erfolgt automatisch via Otsu in sobel_refinement.py

# DINO ↔ Gradient Match (Stufe 6 / SAM3D-Input)
MATCH_CLOSURE_RATIO = 0.65        # Min. Geschlossenheit Kontur auf Gradient-Kante
MATCH_BORDER_TOUCH_RATIO = 0.05   # Max. unerklärter Randkontakt (ohne Gradient)
Z_ALIGN_MIN_KEEP_RATIO = 0.25     # Min. Anteil Segment-Pixel nach Z-Schnitt
MATCH_DEDUP_IOU = 0.10            # IoU-Schwelle bei kleiner Z-Differenz
MATCH_DEDUP_CONTAINMENT = 0.20    # Containment-Schwelle bei kleiner Z-Differenz
MATCH_DEDUP_IOU_FAR = 0.05        # bei mittlerer Z-Differenz
MATCH_DEDUP_CONTAINMENT_FAR = 0.10
MATCH_DEDUP_IOU_DEEP = 0.02       # bei großer Z-Differenz (echte Occlusion)
MATCH_DEDUP_CONTAINMENT_DEEP = 0.05
MATCH_DEDUP_Z_DEEP_M = 0.10       # ab 100 mm = klare Occlusion
MATCH_DEDUP_Z_OCCLUDE_M = 0.02    # ab 20 mm Z-Differenz gilt als occluded
MATCH_DEDUP_Z_DIFF_M = 0.005      # < 5 mm Z-Differenz: gleiche Ebene → beide behalten
MATCH_DEDUP_USE_DINO_BBOX = True  # DINO-BBox zusätzlich zur matched_box prüfen
MATCH_DEDUP_USE_BBOX = True       # Auch BBox-Überlappung prüfen
# MATCH_DEDUP_KEEP_CLOSER siehe Palettenebene-Block (depth_rel)

# Palettenebene (RANSAC, einmal pro Session) + Workspace
WORKSPACE_X_MARGIN_LEFT = 0.15
WORKSPACE_X_MARGIN_RIGHT = 0.15
WORKSPACE_MIN_BOX_OVERLAP = 0.5
PALLET_RANSAC_DISTANCE_M = 0.02
PALLET_RANSAC_ITERATIONS = 1000
PALLET_RANSAC_N = 3
PALLET_RANSAC_STRIDE = 4
PALLET_MIN_INLIER_RATIO = 0.25
PALLET_MAX_NORMAL_ANGLE_DEG = 25.0
PALLET_RANSAC_Y_MIN_RATIO = 0.5
# depth_rel: größer = näher zur Kamera → vordere Kante hat größeren z_plane
MATCH_DEDUP_KEEP_CLOSER = False

# SAM3D (selektiv)
SAM3D_CHALLENGE_THRESHOLD = 0.08   # Nur Masken > 8%
SAM3D_Z_RANGE_THRESHOLD = 0.05     # oder Z-Range > 5cm
SAM3D_DBSCAN_EPS_LARGE = 0.03      # für sehr große Masken
SAM3D_DBSCAN_MIN_SAMPLES_LARGE = 30  # für sehr große Masken

# OpenAI API
OPENAI_API_KEY = "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"  # Hier deinen API-Key eintragen