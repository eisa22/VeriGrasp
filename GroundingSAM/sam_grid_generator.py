"""
GroundingSAM/sam_grid_generator.py
SAM Grid-Prompt Generator: Generiert mehrere SAM-Masken innerhalb einer ROI mittels Grid-Prompts.
"""
import torch
import numpy as np
from PIL import Image
from transformers import SamProcessor, SamModel
from config import *


def calculate_mask_iou(mask1, mask2):
    """Berechnet IoU zwischen zwei binären Masken."""
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    union = np.sum((mask1 > 0) | (mask2 > 0))
    return intersection / union if union > 0 else 0.0


def deduplicate_masks(masks, scores, iou_threshold=0.85):
    """
    Entfernt duplizierte Masken basierend auf IoU.
    
    Args:
        masks: Tensor oder Liste von Masken [N, H, W]
        scores: Tensor oder Liste von Scores [N]
        iou_threshold: IoU-Schwelle für Duplikat-Erkennung
    
    Returns:
        List[np.ndarray]: Liste von deduplizierten Masken
    """
    if len(masks) == 0:
        return []
    
    # Konvertiere zu numpy falls Tensor
    if torch.is_tensor(masks):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = np.array(masks)
    
    if torch.is_tensor(scores):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = np.array(scores)
    
    # Sortiere nach Score (höchste zuerst)
    indices = sorted(range(len(masks_np)), key=lambda i: scores_np[i], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        
        keep.append(i)
        mask_i = masks_np[i]
        
        for j in indices:
            if j == i or j in suppressed or j in keep:
                continue
            
            mask_j = masks_np[j]
            iou = calculate_mask_iou(mask_i, mask_j)
            
            if iou > iou_threshold:
                suppressed.add(j)
    
    return [masks_np[i] for i in keep]


def generate_sam_masks_from_roi(sam_model, sam_processor, image, roi_box, grid_size=None, min_area=None):
    """
    Generiert mehrere SAM-Masken innerhalb einer ROI mittels Grid-Prompts.
    
    Args:
        sam_model: SAM Modell
        sam_processor: SAM Processor
        image: PIL Image
        roi_box: [x1, y1, x2, y2] von DINO
        grid_size: Größe des Grids (default: SAM_GRID_SIZE aus config)
        min_area: Minimale Maskengröße in Pixeln (default: SAM_MASK_MIN_AREA aus config)
    
    Returns:
        List[np.ndarray]: Liste von gefundenen Masken in dieser ROI
    """
    if grid_size is None:
        grid_size = SAM_GRID_SIZE
    if min_area is None:
        min_area = SAM_MASK_MIN_AREA
    
    device = next(sam_model.parameters()).device
    x1, y1, x2, y2 = roi_box
    width, height = x2 - x1, y2 - y1
    
    # Vereinfachter Ansatz: Verwende Sub-Boxen statt Grid-Punkte
    # Teile die ROI in Sub-Boxen auf und verwende Box-Prompts
    sub_boxes = []
    step_x = width / grid_size
    step_y = height / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            sub_x1 = x1 + i * step_x
            sub_y1 = y1 + j * step_y
            sub_x2 = x1 + (i + 1) * step_x
            sub_y2 = y1 + (j + 1) * step_y
            sub_boxes.append([[sub_x1, sub_y1, sub_x2, sub_y2]])
    
    if len(sub_boxes) == 0:
        return []
    
    # Verarbeite Sub-Boxen in Batches
    batch_size = 32  # Anzahl Sub-Boxen pro Batch
    all_masks = []
    all_scores = []
    
    for batch_start in range(0, len(sub_boxes), batch_size):
        batch_end = min(batch_start + batch_size, len(sub_boxes))
        batch_boxes = sub_boxes[batch_start:batch_end]
        
        # Flatten für SAM Processor
        flat_boxes = [box[0] for box in batch_boxes]
        
        try:
            # SAM mit Box-Prompts aufrufen
            sam_inputs = sam_processor(
                image,
                input_boxes=[flat_boxes],
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                sam_outputs = sam_model(**sam_inputs)
            
            # SAM gibt Masken in der Form [num_boxes, num_proposals, H, W]
            batch_masks = sam_outputs.pred_masks
            batch_scores = sam_outputs.iou_scores
            
            # Nimm den besten Vorschlag pro Box (Index 0)
            if len(batch_masks.shape) == 4:
                batch_masks = batch_masks[:, 0, :, :]  # [num_boxes, H, W]
                batch_scores = batch_scores[:, 0]  # [num_boxes]
            
            # Konvertiere zu numpy
            batch_masks_np = batch_masks.cpu().numpy().astype(np.uint8)
            batch_scores_np = batch_scores.cpu().numpy()
            
            # Filtere nach Mindestgröße
            for mask, score in zip(batch_masks_np, batch_scores_np):
                mask_area = np.sum(mask > 0)
                if mask_area >= min_area:
                    all_masks.append(mask)
                    all_scores.append(score)
        
        except Exception as e:
            print(f"      [WARN] Fehler bei Batch {batch_start}-{batch_end}: {e}")
            continue
    
    if len(all_masks) == 0:
        return []
    
    # Dedupliziere Masken
    unique_masks = deduplicate_masks(
        all_masks, 
        all_scores, 
        iou_threshold=SAM_DEDUPLICATION_IOU
    )
    
    return unique_masks


def generate_sam_masks_for_all_rois(session_path, boxes, labels, sam_model=None, sam_processor=None):
    """
    Generiert SAM-Masken für alle ROIs mittels Grid-Prompts.
    
    Args:
        session_path: Pfad zur Session
        boxes: Liste von Boxen [x1, y1, x2, y2]
        labels: Liste von Labels
        sam_model: Optional, SAM Modell (wird geladen falls None)
        sam_processor: Optional, SAM Processor (wird geladen falls None)
    
    Returns:
        tuple: (masks, boxes, scores, labels) - kann mehr Masken sein als Boxen!
    """
    from path_utils import get_rgb_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Lade Modelle falls nicht übergeben
    if sam_model is None or sam_processor is None:
        from transformers import SamProcessor, SamModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
        sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    
    # Lade Bild
    image = Image.open(get_rgb_path(session_path)).convert("RGB")
    
    all_masks = []
    all_boxes = []
    all_labels = []
    all_scores = []
    
    print(f"\n{'='*60}")
    print(f"SAM Grid-Prompts - Segmentierung in ROIs")
    print(f"{'='*60}")
    print(f"Verarbeite {len(boxes)} ROIs mit Grid-Größe {SAM_GRID_SIZE}x{SAM_GRID_SIZE}")
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        print(f"\n[ROI {i+1}/{len(boxes)}] '{label}': Box={box}")
        
        roi_masks = generate_sam_masks_from_roi(
            sam_model, sam_processor, image, box,
            grid_size=SAM_GRID_SIZE,
            min_area=SAM_MASK_MIN_AREA
        )
        
        print(f"  → {len(roi_masks)} Masken gefunden")
        
        # Für jede gefundene Maske: Box berechnen und hinzufügen
        for j, mask in enumerate(roi_masks):
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            
            new_box = [float(xs.min()), float(ys.min()), 
                      float(xs.max()), float(ys.max())]
            
            all_masks.append(mask)
            all_boxes.append(new_box)
            all_labels.append(label)
            all_scores.append(1.0)  # Placeholder Score
    
    print(f"\n[SAM Grid] Gesamt: {len(all_masks)} Masken aus {len(boxes)} ROIs")
    print(f"{'='*60}\n")
    
    return all_masks, all_boxes, all_scores, all_labels

