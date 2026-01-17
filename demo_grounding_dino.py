"""
Demo: Grounding DINO + SAM für Objekterkennung und -trennung
Zeigt die erkannten Objekte im RGB-Bild mit Masken und Bounding Boxes
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
import os
from GroundingSAM.grounding_sam import run_grounding_sam
from path_utils import get_all_session_paths

def visualize_results(image_path, boxes, masks, labels, scores):
    """
    Visualisiert die erkannten Objekte mit farbigen Masken und Bounding Boxes
    """
    # Lade Originalbild
    orig_image = Image.open(image_path).convert("RGB")
    img_draw = orig_image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Generiere unterschiedliche Farben für jedes Objekt
    rng = np.random.default_rng(seed=42)
    colors = [tuple(rng.integers(80, 255, size=3).tolist()) for _ in range(len(masks))]
    
    print(f"\n{'='*60}")
    print(f"ERKANNTE OBJEKTE: {len(masks)}")
    print(f"{'='*60}")
    
    for i, (box, mask, label, score) in enumerate(zip(boxes, masks, labels, scores)):
        c = colors[i]
        score_val = score.item() if torch.is_tensor(score) else score
        
        # Bounding Box zeichnen
        x0, y0, x1, y1 = [int(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=c, width=4)
        
        # Label und Score
        text = f"{i+1}: {label} ({score_val:.2f})"
        draw.text((x0, max(0, y0-16)), text, fill=c)
        
        # Semi-transparente Maske
        color_layer = np.zeros((*mask.shape, 3), dtype=np.uint8)
        color_layer[...] = c
        alpha = (mask * 120).astype(np.uint8)
        img_draw.paste(Image.fromarray(color_layer), mask=Image.fromarray(alpha))
        
        # Statistiken
        mask_pixels = np.sum(mask)
        print(f"Objekt {i+1}: '{label}'")
        print(f"  Score: {score_val:.3f}")
        print(f"  Größe: {mask_pixels} Pixel")
        print(f"  Box: [{x0}, {y0}] → [{x1}, {y1}]")
        print()
    
    return img_draw


def main():
    print(f"{'='*60}")
    print("GROUNDING DINO + SAM OBJEKTTRENNUNG")
    print(f"{'='*60}")
    
    # Hole erste Session
    all_sessions = get_all_session_paths()
    
    if len(all_sessions) == 0:
        print("Keine Sessions gefunden!")
        return
    
    # Verarbeite erste Session
    session_path = all_sessions[0]
    session_name = os.path.basename(session_path)
    image_path = os.path.join(session_path, "rgb", "rgb_0000.png")
    
    print(f"\nSession: {session_name}")
    print(f"Bild: {image_path}")
    
    # Führe Grounding DINO + SAM aus
    boxes, masks, scores, labels = run_grounding_sam(session_path)
    
    if len(masks) == 0:
        print("\n❌ Keine Objekte erkannt!")
        return
    
    # Visualisiere Ergebnisse
    result_image = visualize_results(image_path, boxes, masks, labels, scores)
    
    # Zeige Bild
    print(f"\n{'='*60}")
    print("VISUALISIERUNG")
    print(f"{'='*60}")
    print(f"✓ {len(masks)} Objekte erkannt und getrennt")
    print(f"Zeige Ergebnis...")
    
    result_image.show(title=f"Grounding DINO + SAM: {session_name}")
    
    # Optional: Speichere Ergebnis
    output_path = os.path.join(session_path, "grounding_dino_result.png")
    result_image.save(output_path)
    print(f"\n✓ Ergebnis gespeichert: {output_path}")


if __name__ == "__main__":
    main()



