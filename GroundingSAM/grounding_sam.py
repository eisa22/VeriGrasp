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

def run_grounding_sam(session_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Modelle laden
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)

    sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)

    # Bild laden
    orig_image = Image.open(get_rgb_path(session_path)).convert("RGB")
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

    print(f"\n{'='*60}")
    print(f"GROUNDING DINO - Objekterkennung")
    print(f"{'='*60}")
    print(f"Text Prompts: {TEXT_PROMPT}")
    print(f"BOX_THRESHOLD: {BOX_THRESHOLD} (Box Confidence)")
    print(f"TEXT_THRESHOLD: {TEXT_THRESHOLD} (Text-Matching)")
    print(f"Erkannte Objekte: {len(boxes)}")
    print(f"")
    
    for i in range(len(boxes)):
        score_val = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]
        label_str = labels[i] if i < len(labels) else "N/A"
        print(f"[DINO] Box {i}: Label='{label_str}', Score={score_val:.3f}")

    if len(boxes) == 0:
        return [], [], [], []

    # --- Filter: Entferne Boxen ohne Label ---
    # Wenn TEXT_THRESHOLD hoch ist, filtert DINO die Labels aber behält die Boxen
    # Wir filtern manuell Boxen ohne gültiges Label
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []
    
    for i in range(len(boxes)):
        label = labels[i] if i < len(labels) else ""
        if label and label.strip():  # Nur Boxen mit nicht-leerem Label
            filtered_boxes.append(boxes[i])
            filtered_labels.append(label)
            filtered_scores.append(scores[i])
        else:
            score_val = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]
            print(f"[FILTER] Box {i} entfernt: Label leer (Score={score_val:.3f})")
    
    boxes = filtered_boxes
    labels = filtered_labels
    scores = filtered_scores
    
    print(f"\n[FILTER] Nach Label-Filter: {len(boxes)} Boxen übrig")
    
    if len(boxes) == 0:
        print("[FILTER] Keine Boxen mit gültigen Labels → Abbruch")
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

    print(f"\n{'='*60}")
    print(f"SAM - Segmentierung")
    print(f"{'='*60}")
    print(f"Masken-Shape: {masks.shape}")
    print(f"SAM erzeugt {masks.shape[1]} Masken-Vorschläge pro Box")
    print(f"→ Wir verwenden jeweils den besten Vorschlag (Index 0)")
    
    # SAM gibt Masken in der Form [num_boxes, num_proposals, H, W]
    # num_proposals ist typischerweise 3 (3 Qualitätsstufen pro Box)
    # Wir nehmen nur den besten Vorschlag (Index 0) pro Box
    num_boxes = masks.shape[0]
    print(f"Verarbeite {num_boxes} Boxen von DINO\n")

    cleaned_masks = []
    cleaned_boxes = []
    cleaned_scores = []
    cleaned_labels = []

    for i in range(num_boxes):
        # Nehme den besten Masken-Vorschlag (Index 0 in Dimension 1)
        m = masks[i, 0].cpu().numpy().astype(np.uint8)
        
        score_val = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]
        mask_pixels = m.sum()
        
        print(f"[SAM] Box {i}: Label='{labels[i]}', Score={score_val:.3f}, Maske={mask_pixels} Pixel")
        
        if mask_pixels < 200:
            print(f"      → Übersprungen (Maske zu klein)")
            continue

        cleaned_masks.append(m)
        cleaned_boxes.append(boxes[i])
        cleaned_scores.append(scores[i])
        cleaned_labels.append(labels[i])
        print(f"      → Akzeptiert ✓")

    print(f"\n{'='*60}")
    print(f"GROUNDING SAM Zusammenfassung")
    print(f"{'='*60}")
    print(f"DINO erkannte: {len(boxes)} Boxen")
    print(f"SAM segmentierte: {num_boxes} Masken")
    print(f"Nach Filter: {len(cleaned_masks)} Masken")
    print(f"→ Diese {len(cleaned_masks)} Masken gehen an SAM3D")
    print(f"{'='*60}\n")

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
