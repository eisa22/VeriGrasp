"""
Segmentation/filtering.py
Modul zum Filtern von überlappenden und duplizierten Masken.
"""
import numpy as np
import torch


def filter_overlapping_masks(boxes, masks, scores, labels):
    """
    Filtert überlappende Masken:
    - Entfernt große Masken die mit kleineren überlappen
    - Entfernt fast identische Masken (Duplikate)
    
    Returns:
        tuple: (gefilterte_boxes, gefilterte_masks, gefilterte_scores, gefilterte_labels)
    """
    n = len(masks)
    if n <= 1:
        return boxes, masks, scores, labels
    
    # Berechne Maskengrößen
    mask_sizes = [np.sum(m) for m in masks]
    
    # Markiere Masken zum Entfernen
    to_remove = set()
    
    for i in range(n):
        if i in to_remove:
            continue
            
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            
            # Berechne Überlappung
            overlap = np.sum((masks[i] > 0) & (masks[j] > 0))
            
            if overlap == 0:
                continue
            
            # IoU berechnen
            union = np.sum((masks[i] > 0) | (masks[j] > 0))
            iou = overlap / union if union > 0 else 0
            
            # Fast identische Masken (IoU > 0.9) → entferne die mit niedrigerem Score
            if iou > 0.9:
                score_i = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]
                score_j = scores[j].item() if torch.is_tensor(scores[j]) else scores[j]
                if score_i >= score_j:
                    to_remove.add(j)
                    print(f"  [FILTER] Maske {j} '{labels[j]}' entfernt: Duplikat von {i} (IoU={iou:.2f})")
                else:
                    to_remove.add(i)
                    print(f"  [FILTER] Maske {i} '{labels[i]}' entfernt: Duplikat von {j} (IoU={iou:.2f})")
                continue
    
    # Spezialfall: Sehr große Masken (> 30% des Bildes) die mit mehreren kleinen überlappen
    total_pixels = masks[0].shape[0] * masks[0].shape[1]
    for i in range(n):
        if i in to_remove:
            continue
        
        if mask_sizes[i] > 0.3 * total_pixels:
            overlap_count = 0
            for j in range(n):
                if i == j or j in to_remove:
                    continue
                overlap = np.sum((masks[i] > 0) & (masks[j] > 0))
                if overlap > 0.5 * mask_sizes[j]:
                    overlap_count += 1
            
            if overlap_count >= 2:
                to_remove.add(i)
                print(f"  [FILTER] Maske {i} '{labels[i]}' entfernt: Große Maske ({mask_sizes[i]} Pixel) überlappt mit {overlap_count} kleineren")
    
    # Gefilterte Listen erstellen
    filtered_boxes = [b for idx, b in enumerate(boxes) if idx not in to_remove]
    filtered_masks = [m for idx, m in enumerate(masks) if idx not in to_remove]
    filtered_scores = [s for idx, s in enumerate(scores) if idx not in to_remove]
    filtered_labels = [l for idx, l in enumerate(labels) if idx not in to_remove]
    
    return filtered_boxes, filtered_masks, filtered_scores, filtered_labels
