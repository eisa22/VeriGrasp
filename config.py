# config.py

# config.py

# Basis-Pfad zu deiner Datenquelle (Ordner der Session)
BASE_PATH = "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_07"

# Session-Pfad zeigt standardmäßig auf denselben Ordner
SESSION_PATH = BASE_PATH  

# Debug-Modus für Visualisierung
DEBUG = True

# DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
DINO_MODEL_ID = "/home/samuel/Thesis/VisionPipeline/LocalModels/GroundingDINO"
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
MATCH_DEDUP_KEEP_CLOSER = True    # True = kleinere Tiefe gewinnt (vordere/obere Kante)

# SAM3D (selektiv)
SAM3D_CHALLENGE_THRESHOLD = 0.08   # Nur Masken > 8%
SAM3D_Z_RANGE_THRESHOLD = 0.05     # oder Z-Range > 5cm
SAM3D_DBSCAN_EPS_LARGE = 0.03      # für sehr große Masken
SAM3D_DBSCAN_MIN_SAMPLES_LARGE = 30  # für sehr große Masken

# OpenAI API
OPENAI_API_KEY = "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"  # Hier deinen API-Key eintragen