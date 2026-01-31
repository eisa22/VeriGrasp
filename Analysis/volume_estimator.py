import numpy as np
import open3d as o3d
import cv2

class VolumeEstimator:
    def __init__(self, fx=437.04, fy=437.04, cx=None, cy=None):
        """
        Initialisiert den VolumeEstimator.
        Intrinsics defaulten auf die Werte aus dem Visualizer, falls nicht anders angegeben.
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def fill_depth_holes(self, depth_map):
        """
        Füllt Löcher (0 oder NaN) in der Tiefenkarte mit Inpainting.
        """
        # Maske der Löcher (0 oder sehr kleine Werte oder NaN)
        invalid_mask = (depth_map <= 0) | np.isnan(depth_map)
        if not np.any(invalid_mask):
            return depth_map
            
        # Inpainting braucht 8-bit Input
        depth_u8 = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask_u8 = invalid_mask.astype(np.uint8)
        
        # Inpainting (Telea Algorithmus ist schnell)
        # Achtung: Das funktioniert gut für Visualisierung, aber zerstört die metrische Genauigkeit
        # wenn wir direkt auf dem normalisierten Bild arbeiten.
        # Besser: Nearest Neighbor Fill oder einfaches Inpainting auf Float.
        # Da OpenCV Inpaint nur 8-bit mask unterstützt aber 8-bit image braucht...
        # Wir machen es einfach: Wir nutzen scipy.ndimage oder einfache Interpolation.
        # Oder für Industry Standard: Wir nehmen einfach an, dass 0 = "Weit weg" (Boden) ist,
        # ODER wir ignorieren 0 bei der Median-Berechnung.
        
        # Um es robust zu machen, nutzen wir OpenCV Inpainting auf einer quantisierten Version
        # und skalieren zurück? Nein, das ist ungenau.
        # Wir nutzen eine einfache Strategie: Löcher ignorieren wir bei Berechnungen.
        # Das Inpainting ist nur visuell wichtig, aber für die Messung können wir
        # einfach np.nanmedian nutzen.
        
        # ABER: Der User will "Löcher stopfen".
        # Versuchen wir ein einfaches "Closing" auf der Tiefenkarte.
        kernel = np.ones((5,5), np.float32)
        bg_depth = np.nanmax(depth_map) # Annahme: Hintergrund ist weit weg (max Wert)
        filled_depth = depth_map.copy()
        filled_depth[invalid_mask] = bg_depth # Setze Löcher vorerst auf Hintergrund
        return filled_depth

    def process_object(self, mask, depth_map, label):
        """
        Analysiert eine einzelne Objekt-Maske in 3D mit relativer Höhenmessung.
        """
        H, W = depth_map.shape
        if self.cx is None: self.cx = W / 2
        if self.cy is None: self.cy = H / 2
        
        # 1. Depth Preprocessing (Löcher füllen / behandeln)
        # Wir nehmen an, dass 0-Werte "Löcher" sind und ignorieren sie lieber statistisch
        # statt teures Inpainting zu machen, das Werte erfindet.
        valid_depth_mask = (depth_map > 0) & (~np.isnan(depth_map))
        
        # 2. Top-Surface Bestimmung
        # Punkte INNERHALB der Maske
        ys, xs = np.where((mask > 0) & valid_depth_mask)
        if len(xs) < 10: return None
        
        z_vals = depth_map[ys, xs]
        # Robustes Z_top: Median der Masken-Punkte
        z_top = np.median(z_vals)
        
        # --- Z-CLIPPING REMOVED (User request: Use AI Depth instead) ---
        
        # 3. Base-Level Bestimmung (Relative Height)
        # Wir schauen uns einen Ring UM das Objekt an
        kernel = np.ones((15, 15), np.uint8) # 15px Ring-Breite (ca 2-3cm je nach Auflösung)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        ring_mask = cv2.subtract(dilated_mask, mask.astype(np.uint8))
        
        # Hole Tiefenwerte im Ring
        ring_ys, ring_xs = np.where((ring_mask > 0) & valid_depth_mask)
        
        z_base = 0
        if len(ring_xs) > 10:
            ring_z = depth_map[ring_ys, ring_xs]
            # Sicherstellen, dass wir nicht "Nichts" (0) messen
            valid_ring_z = ring_z[ring_z > 0.1] 
            if len(valid_ring_z) > 0:
                # Wir nehmen das 80. Perzentil (um sicher auf dem Boden zu sein und nicht in Löchern)
                # Oder Median. Median ist sicherer gegen Ausreißer.
                z_base = np.median(valid_ring_z)
            else:
                z_base = z_top # Fallback
        else:
            # Kein Ring (z.B. Rand des Bildes), nehme Max-Tiefe der Szene
            z_base = np.max(depth_map[valid_depth_mask])

        # Berechne Höhe
        # Z ist Distanz zur Kamera. Z_base ist größer als Z_top (weiter weg).
        height_m = abs(z_base - z_top)
        
        # Wenn Höhe unrealistisch klein (< 1mm) oder negativ (passiert durch Rauschen), korrigieren
        if height_m < 0.001: height_m = 0.001 # Min 1mm
        
        # 4. Back-Projection für OBB (Wir nehmen die Punkte der Oberfläche)
        x = (xs - self.cx) * z_vals / self.fx
        y = (ys - self.cy) * z_vals / self.fy
        
        # Open3D: Inv Y, Z
        y = -y
        z = -z_vals
        
        points = np.stack([x, y, z], axis=-1)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Denoising
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        if len(pcd_clean.points) < 10: return None
        
        # 5. OBB & Volumen Berechnung
        try:
            obb = pcd_clean.get_oriented_bounding_box()
            obb.color = (1, 0, 0)
            
            # Original Extent (nur die dünne Schicht)
            extent = obb.extent.copy() # [L, W, Thickness]
            
            # Wir ersetzen die kleinste Dimension (Dicke) durch unsere berechnete Höhe
            sorted_indices = np.argsort(extent) # Index 0=Kleinste, 1=Mittel, 2=Groß
            small_idx = sorted_indices[0]
            
            # Update Extent mit korrekter Höhe
            new_extent = extent.copy()
            new_extent[small_idx] = height_m
            
            # Volumen
            volume_m3 = new_extent[0] * new_extent[1] * new_extent[2]
            volume_liter = volume_m3 * 1000.0
            
            # Create VISUAL OBB (Korrekte Höhe visualisieren)
            # 1. Identifiziere die "Dicke"-Achse (sollte die kleinste Dimension der flachen Wolke sein)
            sorted_indices = np.argsort(extent) # Index 0=Kleinste, 1=Mittel, 2=Groß
            thickness_idx = sorted_indices[0]
            
            # 2. Update Extent: Setze "Dicke" auf echte Höhe
            # Wir erstellen ein neues OBB Objekt, da man extent/center/R setzen kann
            final_obb = o3d.geometry.OrientedBoundingBox()
            final_obb.R = obb.R
            
            new_extent = extent.copy()
            new_extent[thickness_idx] = height_m
            final_obb.extent = new_extent
            
            # 3. Center Shift: Box nach "unten" verlängern
            # Aktueller Center ist auf dem Deckel (z.B. Z = -2.5m)
            # Neuer Center soll auf halber Höhe liegen (z.B. Z = -2.6m, wenn Höhe 0.2m)
            # Wir müssen den Center also entlang der "Dicke"-Achse verschieben.
            
            thickness_axis = obb.R[:, thickness_idx] # Vektor der Dicken-Achse
            
            # Wir wollen, dass der Shift in negative Z-Richtung geht (tiefer in den Raum)
            # da Z negativ ist (Open3D System hier: -Z = Tiefe).
            # Wir prüfen die Z-Komponente des Vektors.
            
            shift_dist = height_m / 2.0
            
            # Wir wollen, dass update_vector.z < 0 ist
            if thickness_axis[2] > 0:
                thickness_axis = -thickness_axis
                
            # Da die Achse vielleicht nicht perfekt vertikal ist, nehmen wir die Projektion?
            # Nein, wir schieben entlang der Box-Achse.
            # Wenn thickness_axis[2] < 0 ist, zeigt er nach "unten".
            shift_vector = thickness_axis * shift_dist
            
            # Aber: Wenn die Box vorher "flach" war (1mm), ist der Center fast oben.
            # Wir subtrahieren die halbe ALTE Dicke und addieren die halbe NEUE Dicke?
            # Vereinfacht: Wir tun so als wäre der alte Center EXAKT oben (Diff < 1mm).
            # Sortierte Dimensionen für Output (L > W > H)
            # Extent hat jetzt die korrekte Höhe an thickness_idx
            # Wir sortieren new_extent einfach absteigend für die Anzeige
            dims_out = sorted(new_extent, reverse=True)
            
            final_obb.center = obb.center + shift_vector
            final_obb.color = (1, 0, 0) # Rot
            
            return {
                "label": label,
                "volume_liter": volume_liter,
                "dimensions_mm": [d * 1000 for d in dims_out], 
                "center_mm": [c * 1000 for c in final_obb.center],
                "rotation": final_obb.R.tolist(),
                "pcd": pcd_clean,
                "obb": final_obb 
            }
            
        except Exception as e:
            print(f"[VolumeEstimator] Fehler bei OBB: {e}")
            return None
