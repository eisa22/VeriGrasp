# GraspGeneration/suction_net.py

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
import open3d as o3d
from config import DEBUG


@dataclass
class GraspCandidate:
    """
    Repräsentiert einen Greifkandidaten für Sauggreifer.
    
    Attributes:
        position: 3D-Position (x, y, z) des Greifpunkts
        normal: Normale der Greiffläche (normalisiert)
        score: Confidence Score aus dem Netzwerk (0-1)
        quality: Zusätzliche Qualitätsmetrik (0-1)
        object_id: ID des zugehörigen Objekts
    """
    position: np.ndarray  # (3,)
    normal: np.ndarray    # (3,)
    score: float
    quality: float
    object_id: int = 0
    
    def __post_init__(self):
        """Validierung und Normalisierung."""
        self.position = np.array(self.position, dtype=np.float32)
        self.normal = np.array(self.normal, dtype=np.float32)
        # Normalisiere Normale
        norm = np.linalg.norm(self.normal)
        if norm > 0:
            self.normal = self.normal / norm


class SuctionNetWrapper:
    """
    Wrapper für SuctionNet zur Generierung von Sauggreif-Kandidaten.
    
    Basiert auf einem pre-trained Modell (GraspNet-1Billion / Contact-GraspNet).
    Da wir noch kein spezifisches Modell haben, implementieren wir zunächst
    eine geometriebasierte Heuristik, die später durch ein echtes Netzwerk
    ersetzt werden kann.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        score_threshold: float = 0.5,
        num_candidates: int = 10,
        min_points_per_grasp: int = 100
    ):
        """
        Args:
            model_path: Pfad zum pre-trained Modell (optional)
            device: GPU oder CPU
            score_threshold: Minimaler Score für Kandidaten
            num_candidates: Max. Anzahl Kandidaten pro Objekt
            min_points_per_grasp: Minimale Punktanzahl für lokale Analyse
        """
        self.device = device
        self.score_threshold = score_threshold
        self.num_candidates = num_candidates
        self.min_points_per_grasp = min_points_per_grasp
        
        # TODO: Später echtes SuctionNet-Modell laden
        self.model = None
        if model_path:
            self._load_model(model_path)
        
        if DEBUG:
            print(f"[SuctionNet] Initialisiert auf {device}")
            print(f"[SuctionNet] Score Threshold: {score_threshold}")
            print(f"[SuctionNet] Max Candidates: {num_candidates}")
    
    def _load_model(self, model_path: str):
        """Lädt pre-trained SuctionNet Modell."""
        # TODO: Implementierung für echtes Modell
        # z.B. torch.load(model_path) für GraspNet-1Billion
        if DEBUG:
            print(f"[SuctionNet] Lade Modell von {model_path}")
        pass
    
    def predict_grasps(
        self,
        pointcloud: Tuple[np.ndarray, np.ndarray],
        object_id: int = 0
    ) -> List[GraspCandidate]:
        """
        Generiert Greifkandidaten für eine Punktwolke.
        
        Args:
            pointcloud: Tuple von (points, colors) - beide (N, 3) arrays
            object_id: ID des Objekts für Tracking
            
        Returns:
            Liste von GraspCandidate Objekten, sortiert nach Score
        """
        points, colors = pointcloud
        
        if len(points) < self.min_points_per_grasp:
            if DEBUG:
                print(f"[SuctionNet] Objekt {object_id}: Zu wenige Punkte ({len(points)})")
            return []
        
        # Wenn echtes Modell verfügbar, nutze es
        if self.model is not None:
            return self._predict_with_model(points, colors, object_id)
        
        # Ansonsten: Geometrie-basierte Heuristik
        return self._predict_with_heuristic(points, colors, object_id)
    
    def _predict_with_model(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        object_id: int
    ) -> List[GraspCandidate]:
        """Inferenz mit pre-trained SuctionNet."""
        # TODO: Implementierung für echtes Netzwerk
        # - Punktwolke normalisieren/samplen
        # - Durch Netzwerk laufen lassen
        # - Ausgabe in GraspCandidate konvertieren
        pass
    
    def _predict_with_heuristic(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        object_id: int
    ) -> List[GraspCandidate]:
        """
        Geometriebasierte Heuristik für Greifkandidaten.
        
        Strategie:
        1. Schätze Normalen für alle Punkte
        2. Finde horizontale/nach-oben-gerichtete Flächen
        3. Bevorzuge Punkte in der Mitte des Objekts (Stabilitätsheuristik)
        4. Ranke nach Planarity und Position
        """
        # Open3D Punktwolke für Normalen-Schätzung
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Normalen schätzen
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05,  # 5cm Radius
                max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(30)
        
        normals = np.asarray(pcd.normals)
        
        # Filtere nach Normale (bevorzuge nach oben gerichtete Flächen)
        # Z-Komponente sollte positiv sein (nach oben)
        z_component = normals[:, 2]
        
        # Kandidaten: Punkte mit z-normal > 0.7 (ca. 45° zur Vertikalen)
        candidate_mask = z_component > 0.7
        candidate_indices = np.where(candidate_mask)[0]
        
        if len(candidate_indices) == 0:
            if DEBUG:
                print(f"[SuctionNet] Objekt {object_id}: Keine geeigneten Flächen gefunden")
            return []
        
        # Berechne Scores basierend auf:
        # 1. Z-Komponente der Normale (je höher, desto besser)
        # 2. Höhe des Punkts (höhere Punkte bevorzugen)
        # 3. Distanz zum Centroid (mittlere Punkte bevorzugen)
        
        candidate_points = points[candidate_indices]
        candidate_normals = normals[candidate_indices]
        
        # Score-Komponenten
        z_scores = candidate_normals[:, 2]  # 0-1
        
        # Höhen-Score (normalisiert)
        heights = candidate_points[:, 2]
        height_scores = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
        
        # Centroid-Distanz (invertiert und normalisiert)
        centroid = candidate_points.mean(axis=0)
        distances = np.linalg.norm(candidate_points - centroid, axis=1)
        dist_scores = 1.0 - (distances / (distances.max() + 1e-6))
        
        # Kombinierter Score (gewichtet)
        combined_scores = (
            0.4 * z_scores +
            0.3 * height_scores +
            0.3 * dist_scores
        )
        
        # Qualitätsmetrik: Lokale Planarity
        qualities = self._compute_local_planarity(pcd, candidate_indices)
        
        # Erstelle GraspCandidates
        candidates = []
        for idx, orig_idx in enumerate(candidate_indices):
            candidates.append(GraspCandidate(
                position=points[orig_idx],
                normal=normals[orig_idx],
                score=float(combined_scores[idx]),
                quality=float(qualities[idx]),
                object_id=object_id
            ))
        
        # Sortiere nach Score (absteigend)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Filtere nach Threshold und limitiere Anzahl
        candidates = [
            c for c in candidates
            if c.score >= self.score_threshold
        ][:self.num_candidates]
        
        if DEBUG:
            print(f"[SuctionNet] Objekt {object_id}: {len(candidates)} Kandidaten generiert")
        
        return candidates
    
    def _compute_local_planarity(
        self,
        pcd: o3d.geometry.PointCloud,
        indices: np.ndarray,
        radius: float = 0.03
    ) -> np.ndarray:
        """
        Berechnet lokale Planarity für jeden Punkt.
        
        Planarity = 1 - (kleinster Eigenwert / Summe Eigenwerte)
        Je näher an 1, desto planarer die Umgebung.
        """
        points = np.asarray(pcd.points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        planarities = np.zeros(len(indices))
        
        for i, idx in enumerate(indices):
            # Finde Nachbarn
            [k, neighbor_indices, _] = tree.search_radius_vector_3d(
                points[idx], radius
            )
            
            if k < 3:
                planarities[i] = 0.0
                continue
            
            # Kovarianzmatrix der Nachbarn
            neighbors = points[neighbor_indices]
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)
            
            # Eigenwerte
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Absteigend
            
            # Planarity: (ev1 - ev2) / ev0
            if eigenvalues[0] > 1e-6:
                planarity = 1.0 - (eigenvalues[2] / eigenvalues.sum())
            else:
                planarity = 0.0
            
            planarities[i] = np.clip(planarity, 0.0, 1.0)
        
        return planarities

