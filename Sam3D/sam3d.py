"""
Sam3D/sam3d.py
3D-Verfeinerung: Challengt große SAM-Masken und teilt sie mit 3D-Clustering auf.
Selektives Splitting: Nur Masken > Threshold ODER mit großen Z-Range werden gesplittet.
"""
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
from sklearn.cluster import DBSCAN
import os
from config import *


def refine_masks_3d(masks, boxes, scores, labels, session_path):
    """
    Challengt ALLE Masken und sucht nach Sub-Objekten mit 3D-Clustering.
    Trennt insbesondere Pakete auf unterschiedlichen Z-Ebenen.
    
    Args:
        masks: Liste von binären 2D-Masken (von SAM)
        boxes, scores, labels: Zugehörige Daten
        session_path: Pfad zur Session
        
    Returns:
        tuple: (verfeinerte_masks, boxes, scores, labels) - kann mehr Masken sein!
    """
    print(f"\n[SAM3D] Starte 3D-Verfeinerung von {len(masks)} Masken...")
    print(f"[SAM3D] Strategie: Selektives Splitting (nur große/mehrschichtige Masken)")
    print(f"[SAM3D] Challenge-Threshold: {SAM3D_CHALLENGE_THRESHOLD*100:.1f}% Bildfläche oder Z-Range > {SAM3D_Z_RANGE_THRESHOLD*1000:.0f}mm")
    
    # Lade RGB und Depth
    rgb_path = os.path.join(session_path, "rgb", "rgb_0000.png")
    depth_path = os.path.join(session_path, "distance_to_image_plane", 
                               "distance_to_image_plane_0000.npy")
    
    rgb = np.array(Image.open(rgb_path))[:, :, :3]
    depth = np.load(depth_path)
    H, W = depth.shape
    total_pixels = H * W
    
    # Kamera-Intrinsics
    fx = fy = 437.04
    cx, cy = W / 2, H / 2
    
    # Erstelle 3D-Punktwolke
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    all_points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    refined_masks = []
    refined_boxes = []
    refined_scores = []
    refined_labels = []
    
    for i, (mask, box, score, label) in enumerate(zip(masks, boxes, scores, labels)):
        mask_np = np.asarray(mask)
        mask_size = mask_np.sum()
        mask_ratio = mask_size / total_pixels
        
        print(f"\n  [SAM3D] Maske {i} '{label}': {mask_size} Pixel ({mask_ratio*100:.1f}%)")
        
        # Extrahiere 3D-Punkte
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0:
            print(f"    → Keine Pixel, überspringe")
            continue
        
        linear_idx = ys * W + xs
        points_3d = all_points_3d[linear_idx]
        
        # Filtere ungültige Punkte
        valid_depth = depth[ys, xs] > 0
        if valid_depth.sum() < 200:
            print(f"    → Zu wenig gültige 3D-Punkte ({valid_depth.sum()}), behalte Original")
            refined_masks.append(mask)
            refined_boxes.append(box)
            refined_scores.append(score)
            refined_labels.append(label)
            continue
        
        points_3d_valid = points_3d[valid_depth]
        xs_valid = xs[valid_depth]
        ys_valid = ys[valid_depth]
        
        # -------------------------------------------------------------------------
        # STRATEGIE 1: Analysiere Z-Ebenen (Höhe)
        # -------------------------------------------------------------------------
        z_coords = points_3d_valid[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        z_range = z_max - z_min
        
        print(f"    → Z-Range: {z_range*1000:.1f}mm (von {z_min*1000:.0f}mm bis {z_max*1000:.0f}mm)")
        
        # -------------------------------------------------------------------------
        # Entscheidung: Soll diese Maske gesplittet werden?
        # -------------------------------------------------------------------------
        # Selektives Splitting: Nur wenn:
        # 1. Maske > Challenge-Threshold ODER
        # 2. Z-Range > Z-Range-Threshold
        should_challenge = (mask_ratio > SAM3D_CHALLENGE_THRESHOLD) or (z_range > SAM3D_Z_RANGE_THRESHOLD)
        
        if not should_challenge:
            print(f"    → Maske zu klein und flach, behalte Original (kein Splitting)")
            refined_masks.append(mask)
            refined_boxes.append(box)
            refined_scores.append(score)
            refined_labels.append(label)
            continue
        
        # Wenn Z-Range > Threshold → wahrscheinlich mehrere Ebenen
        multiple_z_levels = z_range > SAM3D_Z_RANGE_THRESHOLD
        
        # -------------------------------------------------------------------------
        # STRATEGIE: 3D-Clustering mit DBSCAN
        # -------------------------------------------------------------------------
        # Adaptive Parameter basierend auf Maskengröße
        if mask_ratio > 0.15:  # sehr große Masken
            eps = SAM3D_DBSCAN_EPS_LARGE  # 3cm statt 4cm
            min_samples = SAM3D_DBSCAN_MIN_SAMPLES_LARGE  # 30 statt 50
        else:
            eps = 0.04  # Standard: 4cm
            min_samples = 50  # Standard: 50
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_3d_valid)
        labels_cluster = clustering.labels_
        
        unique_labels = set(labels_cluster)
        unique_labels.discard(-1)  # Entferne Noise
        
        n_clusters = len(unique_labels)
        
        print(f"    → DBSCAN (eps={eps*1000:.0f}mm, min_samples={min_samples}) fand {n_clusters} Cluster (Noise: {(labels_cluster == -1).sum()} Punkte)")
        
        if multiple_z_levels:
            print(f"    → Warnung: Mehrere Z-Ebenen erkannt! (Range: {z_range*1000:.1f}mm)")
        
        # -------------------------------------------------------------------------
        # Entscheidung: Splitting oder nicht?
        # -------------------------------------------------------------------------
        # Splitting wenn mehrere Cluster gefunden
        should_split = n_clusters > 1
        
        if should_split:
            print(f"    → ✓ Splitting durchführen: {n_clusters} Sub-Objekte")
            
            # Erstelle separate Masken für jedes Cluster
            for cluster_id in unique_labels:
                cluster_mask = np.zeros((H, W), dtype=np.uint8)
                cluster_indices = labels_cluster == cluster_id
                
                xs_cluster = xs_valid[cluster_indices]
                ys_cluster = ys_valid[cluster_indices]
                
                # Minimale Größe: 300 Pixel (reduziert von 500)
                if len(xs_cluster) < 300:
                    print(f"      → Cluster {cluster_id} zu klein ({len(xs_cluster)} Pixel), überspringe")
                    continue
                
                cluster_mask[ys_cluster, xs_cluster] = 1
                
                # Morphologische Glättung
                kernel = np.ones((5, 5), np.uint8)
                cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
                cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                
                # Berechne neue Box
                ys_new, xs_new = np.where(cluster_mask > 0)
                if len(xs_new) == 0:
                    continue
                
                # Berechne Z-Stats für dieses Cluster
                cluster_points_3d = points_3d_valid[cluster_indices]
                z_mean = cluster_points_3d[:, 2].mean()
                
                new_box = [float(xs_new.min()), float(ys_new.min()), 
                           float(xs_new.max()), float(ys_new.max())]
                
                refined_masks.append(cluster_mask)
                refined_boxes.append(new_box)
                refined_scores.append(score)
                refined_labels.append(f"{label}_obj{cluster_id+1}")
                
                print(f"      → Cluster {cluster_id}: {len(xs_cluster)} Punkte, Z={z_mean*1000:.0f}mm")
            
            print(f"    ✓ Maske erfolgreich in {n_clusters} Sub-Objekte aufgeteilt")
        else:
            # Nur 1 Cluster → Outlier-Removal und behalten
            print(f"    → Nur 1 zusammenhängendes Objekt, behalte mit Outlier-Removal")
            
            # Outlier-Removal
            if len(points_3d_valid) > 100:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_3d_valid)
                pcd_filtered, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                
                if len(inliers) > 200:
                    refined_mask = np.zeros((H, W), dtype=np.uint8)
                    xs_inliers = xs_valid[inliers]
                    ys_inliers = ys_valid[inliers]
                    refined_mask[ys_inliers, xs_inliers] = 1
                    
                    kernel = np.ones((5, 5), np.uint8)
                    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
                    
                    ys_new, xs_new = np.where(refined_mask > 0)
                    if len(xs_new) > 0:
                        new_box = [float(xs_new.min()), float(ys_new.min()), 
                                   float(xs_new.max()), float(ys_new.max())]
                        
                        refined_masks.append(refined_mask)
                        refined_boxes.append(new_box)
                        refined_scores.append(score)
                        refined_labels.append(label)
                    else:
                        refined_masks.append(mask)
                        refined_boxes.append(box)
                        refined_scores.append(score)
                        refined_labels.append(label)
                else:
                    refined_masks.append(mask)
                    refined_boxes.append(box)
                    refined_scores.append(score)
                    refined_labels.append(label)
            else:
                refined_masks.append(mask)
                refined_boxes.append(box)
                refined_scores.append(score)
                refined_labels.append(label)
    
    print(f"\n[SAM3D] Verfeinerung abgeschlossen: {len(masks)} → {len(refined_masks)} Masken")
    delta = len(refined_masks) - len(masks)
    if delta > 0:
        print(f"[SAM3D] ✓ +{delta} zusätzliche Sub-Objekte durch 3D-Analyse gefunden")
    elif delta < 0:
        print(f"[SAM3D] ⚠ {abs(delta)} Masken wurden gefiltert")
    else:
        print(f"[SAM3D] Keine Sub-Objekte gefunden (alle Masken sind einzelne Objekte)")
    
    return refined_masks, refined_boxes, refined_scores, refined_labels
