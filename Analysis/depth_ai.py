import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

class DepthRefiner:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"[DepthAI] Lade Depth Anything V2 (Small) auf {device}...")
        try:
            # Wir nutzen das HF Model. V2 ist oft unter 'depth-anything/Depth-Anything-V2-Small-hf'
            # Falls nicht verfügbar, nutzen wir LiheYoung Version (V1/V2 Base).
            # Hier nutzen wir das performante Small Modell.
            model_id = "LiheYoung/depth-anything-small-hf" 
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
            print("[DepthAI] Modell erfolgreich geladen.")
        except Exception as e:
            print(f"[DepthAI] FEHLER beim Laden des Modells: {e}")
            self.model = None

    def refine_masks(self, rgb_image, masks, labels):
        """
        Nutzt AI-Depth, um Masken zu verfeinern (schneidet überlaufende Teile ab).
        """
        if self.model is None or not masks:
            return masks, labels, None

        # 1. AI Depth Prediction
        inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolieren auf Originalgröße
        H, W = rgb_image.size[::-1] # PIL ist W, H
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )
        depth_map = prediction.squeeze().cpu().numpy()
        
        # Normalisieren für Edge Detection (0-255)
        depth_u8 = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 2. AI-Kanten erkennen (auf der scharfen AI-Tiefenkarte)
        # Depth Anything liefert sehr scharfe Kanten. Wir nutzen Canny oder Sobel darauf.
        # Da wir "Leck" verhindern wollen, nehmen wir starke Kanten.
        edges = cv2.Canny(depth_u8, 50, 150)
        
        # Dilate Edges slightly to separate objects cleanly
        kernel = np.ones((3,3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        refined_mask_list = []
        refined_label_list = []
        
        print(f"[DepthAI] Verfeinere {len(masks)} Masken mit AI-Depth...")
        
        for i, (mask, label) in enumerate(zip(masks, labels)):
            # Taktik: Maske MINUS AI-Kanten
            # Wenn das Objekt über eine Kante "leckt", wird es hier getrennt.
            
            # Maske zersägen
            cut_mask = mask.copy()
            cut_mask[edges_dilated > 0] = 0 # Schneide die Kanten raus
            
            # Connected Components: Wir behalten nur das größte Stück (den Hauptkörper)
            # Das kleine "Leck" auf dem unteren Paket sollte jetzt isoliert sein.
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(cut_mask.astype(np.uint8))
            
            if num_labels > 2: # 0 ist Background, 1..N sind Teile. Wenn >2 teile, gab es einen Cut.
                # Wähle den größten Teil
                max_area = 0
                best_label = 1
                for l_idx in range(1, num_labels):
                    area = stats[l_idx, cv2.CC_STAT_AREA]
                    if area > max_area:
                        max_area = area
                        best_label = l_idx
                
                final_mask = (labels_im == best_label).astype(np.uint8)
                
                # Optional: Schließe die Löcher, die durch Canny entstanden sind (Closing)
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                # Wenn wir zu viel verloren haben (>50%), war der Cut vielleicht falsch?
                # Hier vertrauen wir der AI: Besser zu klein als falsch.
                refined_mask_list.append(final_mask)
                refined_label_list.append(label)
            else:
                # Wurde nicht getrennt (glatt), behalte Original (bzw. die Version ohne Kantenpixel)
                # Wir nehmen lieber die cleaned version (ohne edges), für scharfe Ränder.
                refined_mask_list.append(cut_mask.astype(np.uint8))
                refined_label_list.append(label)

        return refined_mask_list, refined_label_list, depth_u8
