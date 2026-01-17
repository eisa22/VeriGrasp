# config.py

# config.py

# Basis-Pfad zu deiner Datenquelle (Ordner der Session)
BASE_PATH = "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_07"

# Session-Pfad zeigt standardmäßig auf denselben Ordner
SESSION_PATH = BASE_PATH  

# Debug-Modus für Visualisierung
DEBUG = True

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
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

# SAM3D (selektiv)
SAM3D_CHALLENGE_THRESHOLD = 0.08   # Nur Masken > 8%
SAM3D_Z_RANGE_THRESHOLD = 0.05     # oder Z-Range > 5cm
SAM3D_DBSCAN_EPS_LARGE = 0.03      # für sehr große Masken
SAM3D_DBSCAN_MIN_SAMPLES_LARGE = 30  # für sehr große Masken


