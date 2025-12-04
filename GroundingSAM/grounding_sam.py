# grounding_sam.py

import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor as DinoProcessor,
    AutoModelForZeroShotObjectDetection as DinoModel,
    SamProcessor,
    SamModel,
)
from config import *
from path_utils import get_rgb_path

def run_grounding_sam():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Modelle laden
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)

    sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)

    # Bild laden
    orig_image = Image.open(get_rgb_path()).convert("RGB")
    resized_image = orig_image.resize((1024, 1024))

    # --- Grounding DINO ---
    inputs = dino_processor(
        images=resized_image,
        text=TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(orig_image.size[1], orig_image.size[0])]
    )

    result = results[0]
    boxes  = result["boxes"].tolist()
    labels = result["text_labels"]
    scores = result["scores"]

    if len(boxes) == 0:
        return [], [], [], []

    # --- SAM ---
    sam_inputs = sam_processor(
        orig_image,
        input_boxes=[boxes],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    low_res_masks = sam_outputs.pred_masks

    mask_results = sam_processor.post_process_masks(
        low_res_masks,
        sam_inputs["original_sizes"].to(device),
        sam_inputs["reshaped_input_sizes"].to(device),
    )

    if isinstance(mask_results, list):
        masks = mask_results[0]
    else:
        masks = mask_results

    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks.unsqueeze(0)

    # SAM kann mehr Masken als Boxen liefern
    num_objects = min(len(boxes), masks.shape[1])

    cleaned_masks = []
    cleaned_boxes = []
    cleaned_scores = []
    cleaned_labels = []

    for i in range(num_objects):
        m = masks[0, i].cpu().numpy().astype(np.uint8)

        if m.sum() < 200:
            continue

        cleaned_masks.append(m)
        cleaned_boxes.append(boxes[i])
        cleaned_scores.append(scores[i])
        cleaned_labels.append(labels[i])

    # --- Debug Visualisierung ---
    if DEBUG:
        img_draw = orig_image.copy()
        draw = ImageDraw.Draw(img_draw)

        rng = np.random.default_rng(seed=42)
        colors = [tuple(rng.integers(80,255,size=3).tolist()) for _ in range(len(cleaned_masks))]

        for i, mask in enumerate(cleaned_masks):
            x0, y0, x1, y1 = cleaned_boxes[i]
            c = colors[i]

            draw.rectangle([x0,y0,x1,y1], outline=c, width=3)
            draw.text((x0, max(0,y0-14)), f"{cleaned_labels[i]} ({cleaned_scores[i]:.2f})", fill=c)

            color_layer = np.zeros((*mask.shape,3), dtype=np.uint8)
            color_layer[...] = c
            alpha = (mask * 120).astype(np.uint8)
            img_draw.paste(Image.fromarray(color_layer), mask=Image.fromarray(alpha))

        img_draw.show()

    return cleaned_boxes, cleaned_masks, cleaned_scores, cleaned_labels
