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

BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.20


