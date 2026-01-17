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


def calculate_box_iou(box1, box2):
    """Berechnet IoU zwischen zwei Boxen [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def filter_large_boxes(boxes, scores, labels, image_width, image_height, max_area_ratio):
    """
    Filtert Boxen die zu groß sind (wahrscheinlich Mehrfachdetektionen).
    
    Args:
        boxes: Liste von Boxen [x1, y1, x2, y2]
        scores, labels: Zugehörige Scores und Labels
        image_width, image_height: Bildabmessungen
        max_area_ratio: Maximaler Anteil der Bildfläche (z.B. 0.12 = 12%)
    
    Returns:
        tuple: (gefilterte_boxes, scores, labels)
    """
    total_area = image_width * image_height
    max_box_area = total_area * max_area_ratio
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        
        if box_area <= max_box_area:
            filtered_boxes.append(box)
            filtered_scores.append(scores[i])
            filtered_labels.append(labels[i])
        else:
            score_val = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]
            print(f"[BOX-FILTER] Box {i} '{labels[i]}' entfernt: Zu groß ({box_area/total_area*100:.1f}% der Bildfläche)")
    
    return filtered_boxes, filtered_scores, filtered_labels


def apply_relative_iou_nms(boxes, scores, labels, iou_threshold=0.3):
    """
    Non-Maximum Suppression mit relativer IoU.
    Behält kleine Boxen, auch wenn sie innerhalb großer liegen.
    
    Args:
        boxes: Liste von Boxen [x1, y1, x2, y2]
        scores: Liste von Scores
        labels: Liste von Labels
        iou_threshold: IoU-Schwelle für NMS
    
    Returns:
        tuple: (gefilterte_boxes, scores, labels)
    """
    if len(boxes) <= 1:
        return boxes, scores, labels
    
    # Konvertiere Scores zu numpy für einfacheres Sortieren
    score_values = []
    for s in scores:
        score_values.append(s.item() if torch.is_tensor(s) else s)
    
    # Sortiere nach Score (höchste zuerst)
    indices = sorted(range(len(boxes)), key=lambda i: score_values[i], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        
        keep.append(i)
        box_i = boxes[i]
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        
        for j in indices:
            if j == i or j in suppressed or j in keep:
                continue
            
            box_j = boxes[j]
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            
            iou = calculate_box_iou(box_i, box_j)
            
            # Relative IoU: Wenn kleine Box in großer Box liegt, weniger aggressiv unterdrücken
            if area_j < area_i:
                # Kleine Box in großer Box: Nur unterdrücken wenn sehr hohe IoU
                relative_threshold = iou_threshold * 1.5
            else:
                # Große Box überlappt kleine: Standard-Threshold
                relative_threshold = iou_threshold
            
            if iou > relative_threshold:
                suppressed.add(j)
                print(f"[NMS] Box {j} '{labels[j]}' unterdrückt: IoU={iou:.3f} mit Box {i}")
    
    filtered_boxes = [boxes[i] for i in keep]
    filtered_scores = [scores[i] for i in keep]
    filtered_labels = [labels[i] for i in keep]
    
    return filtered_boxes, filtered_scores, filtered_labels


def run_grounding_dino_only(session_path: str):
    """
    Führt nur Grounding DINO aus und gibt Boxen zurück (ohne SAM).
    Für Hybrid-Pipeline: DINO liefert grobe Regionen, SAM Grid-Prompts werden separat aufgerufen.
    
    Returns:
        tuple: (boxes, scores, labels, orig_image)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Modelle laden
    dino_processor = DinoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = DinoModel.from_pretrained(DINO_MODEL_ID).to(device)
    
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
    boxes = result["boxes"].tolist()
    labels = result["text_labels"]
    scores = result["scores"]
    
    print(f"\n{'='*60}")
    print(f"GROUNDING DINO - Regionen-Erkennung")
    print(f"{'='*60}")
    print(f"Text Prompts: {TEXT_PROMPT}")
    print(f"BOX_THRESHOLD: {BOX_THRESHOLD} (Box Confidence)")
    print(f"TEXT_THRESHOLD: {TEXT_THRESHOLD} (Text-Matching)")
    print(f"Erkannte Regionen: {len(boxes)}")
    
    # Filter: Entferne Boxen ohne Label
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []
    
    for i in range(len(boxes)):
        label = labels[i] if i < len(labels) else ""
        if label and label.strip():
            filtered_boxes.append(boxes[i])
            filtered_labels.append(label)
            filtered_scores.append(scores[i])
        else:
            score_val = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]
            print(f"[FILTER] Box {i} entfernt: Label leer (Score={score_val:.3f})")
    
    boxes = filtered_boxes
    labels = filtered_labels
    scores = filtered_scores
    
    print(f"[FILTER] Nach Label-Filter: {len(boxes)} Regionen übrig")
    
    if len(boxes) == 0:
        return [], [], [], None
    
    # Box-Größen-Filter
    image_width, image_height = orig_image.size
    boxes, scores, labels = filter_large_boxes(
        boxes, scores, labels, image_width, image_height, MAX_BOX_AREA_RATIO
    )
    print(f"[BOX-FILTER] Nach Größen-Filter: {len(boxes)} Regionen übrig")
    
    # Relative IoU NMS
    boxes, scores, labels = apply_relative_iou_nms(
        boxes, scores, labels, RELATIVE_IOU_NMS_THRESH
    )
    print(f"[NMS] Nach Relative IoU NMS: {len(boxes)} finale Regionen")
    
    return boxes, scores, labels, orig_image


def generate_sam_masks_for_boxes(session_path, boxes, labels, sam_model=None, sam_processor=None):
    """
    Generiert eine SAM-Maske pro DINO-Box.
    
    Args:
        session_path: Pfad zur Session
        boxes: Liste von Boxen [x1, y1, x2, y2] von DINO
        labels: Liste von Labels
        sam_model: Optional, SAM Modell (wird geladen falls None)
        sam_processor: Optional, SAM Processor (wird geladen falls None)
    
    Returns:
        tuple: (masks, boxes, scores, labels)
    """
    import torch
    from path_utils import get_rgb_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Lade Modelle falls nicht übergeben
    if sam_model is None or sam_processor is None:
        sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
        sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    
    # Lade Bild
    orig_image = Image.open(get_rgb_path(session_path)).convert("RGB")
    
    print(f"\n{'='*60}")
    print(f"SAM - Eine Maske pro DINO-Box")
    print(f"{'='*60}")
    print(f"Verarbeite {len(boxes)} Boxen...")
    
    if len(boxes) == 0:
        return [], [], [], []
    
    # SAM mit Box-Prompts aufrufen
    sam_inputs = sam_processor(
        orig_image,
        input_boxes=[boxes],
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)
    
    low_res_masks = sam_outputs.pred_masks
    
    # Handle 5D tensor (batch dimension)
    if len(low_res_masks.shape) == 5:
        low_res_masks = low_res_masks.squeeze(0)
    
    mask_results = sam_processor.post_process_masks(
        low_res_masks.unsqueeze(0) if len(low_res_masks.shape) == 4 else low_res_masks,
        sam_inputs["original_sizes"].to(device),
        sam_inputs["reshaped_input_sizes"].to(device),
    )
    
    if isinstance(mask_results, list):
        masks = mask_results[0]
    else:
        masks = mask_results
    
    # Handle tensor shape
    if len(masks.shape) == 4:
        # [num_boxes, num_proposals, H, W] - take best proposal
        masks = masks[:, 0, :, :]  # [num_boxes, H, W]
    
    print(f"SAM Masken-Shape: {masks.shape}")
    
    cleaned_masks = []
    cleaned_boxes = []
    cleaned_scores = []
    cleaned_labels = []
    
    for i in range(masks.shape[0]):
        m = masks[i].cpu().numpy().astype(np.uint8)
        mask_pixels = m.sum()
        
        print(f"[SAM] Box {i}: Label='{labels[i]}', Maske={mask_pixels} Pixel")
        
        if mask_pixels < 200:
            print(f"      → Übersprungen (Maske zu klein)")
            continue
        
        cleaned_masks.append(m)
        cleaned_boxes.append(boxes[i])
        cleaned_scores.append(1.0)  # Placeholder score
        cleaned_labels.append(labels[i])
        print(f"      → Akzeptiert ✓")
    
    print(f"\n[SAM] {len(cleaned_masks)} Masken generiert aus {len(boxes)} Boxen")
    
    return cleaned_masks, cleaned_boxes, cleaned_scores, cleaned_labels


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

    return cleaned_boxes, cleaned_masks, cleaned_scores, cleaned_labels
