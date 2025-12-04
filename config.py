# config.py

BASE_PATH = "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_07/"

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

DEBUG = True
