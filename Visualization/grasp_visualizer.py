# Visualization/grasp_visualizer.py

import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional

from GraspGeneration.suction_net import GraspCandidate
from Verification.geometric_verifier import ValidatedGrasp


class GraspVisualizer:
    """
    Visualisierung von Greifkandidaten und 3D-Punktwolken.
    
    Features:
    - Farbcodierung nach Validierungsstatus (Grün/Rot/Gelb)
    - Greifnormalen als 3D-Pfeile
    - Mehrere Objekte in einer Szene
    - Heatmap der Greifqualität (optional)
    """
    
    # Farbdefinitionen
    COLOR_PASSED = np.array([0.0, 0.8, 0.0])      # Grün
    COLOR_REJECTED = np.array([0.9, 0.0, 0.0])    # Rot
    COLOR_UNCERTAIN = np.array([0.9, 0.9, 0.0])   # Gelb
    COLOR_BACKGROUND = np.array([0.7, 0.7, 0.7])  # Hellgrau
    
    def __init__(
        self,
        arrow_length: float = 0.05,
        arrow_radius: float = 0.003,
        sphere_radius: float = 0.01,
        show_background: bool = True
    ):
        """
        Args:
            arrow_length: Länge der Normalen-Pfeile (m)
            arrow_radius: Radius der Pfeile (m)
            sphere_radius: Radius der Greifpunkt-Marker (m)
            show_background: Zeige vollständige Szene im Hintergrund
        """
        self.arrow_length = arrow_length
        self.arrow_radius = arrow_radius
        self.sphere_radius = sphere_radius
        self.show_background = show_background
    
    def visualize_grasps(
        self,
        validated_grasps: List[ValidatedGrasp],
        pointclouds: List[Tuple[np.ndarray, np.ndarray]],
        full_scene_pcd: Optional[o3d.geometry.PointCloud] = None,
        window_name: str = "Grasp Visualization"
    ):
        """
        Visualisiert validierte Greifkandidaten in 3D.
        
        Args:
            validated_grasps: Liste von ValidatedGrasp Objekten
            pointclouds: Liste von (points, colors) Tuples pro Objekt
            full_scene_pcd: Optionale Hintergrund-Punktwolke
            window_name: Fenster-Titel
        """
        geometries = []
        
        # 1. Hintergrund-Szene (falls vorhanden)
        if self.show_background and full_scene_pcd is not None:
            bg_pcd = o3d.geometry.PointCloud(full_scene_pcd)
            # Färbe Hintergrund grau
            bg_colors = np.tile(self.COLOR_BACKGROUND, (len(bg_pcd.points), 1))
            bg_pcd.colors = o3d.utility.Vector3dVector(bg_colors)
            geometries.append(bg_pcd)
        
        # 2. Objekt-Punktwolken (farbig hervorgehoben)
        rng = np.random.default_rng(seed=42)
        obj_colors = [
            rng.uniform(0.3, 1.0, size=3)
            for _ in range(len(pointclouds))
        ]
        
        for obj_id, (points, colors) in enumerate(pointclouds):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Verwende zufällige Farbe pro Objekt
            obj_color = obj_colors[obj_id]
            colored = np.tile(obj_color, (len(points), 1))
            pcd.colors = o3d.utility.Vector3dVector(colored)
            
            geometries.append(pcd)
        
        # 3. Greifkandidaten visualisieren
        for grasp in validated_grasps:
            geom = self._create_grasp_geometry(grasp)
            geometries.extend(geom)
        
        # 4. Koordinatensystem (Origin)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)
        
        # 5. Visualisierung öffnen
        print(f"[GraspVisualizer] Öffne {window_name}...")
        print(f"[GraspVisualizer] Anzahl Grasps: {len(validated_grasps)}")
        print(f"  Grün  = Validiert")
        print(f"  Rot   = Abgelehnt")
        print(f"  Gelb  = Unsicher")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1280,
            height=720
        )
    
    def visualize_top_grasps_only(
        self,
        top_grasps: List[ValidatedGrasp],
        pointclouds: List[Tuple[np.ndarray, np.ndarray]],
        window_name: str = "Top Grasps"
    ):
        """
        Zeigt nur die top-validierten Grasps (ohne abgelehnte).
        
        Args:
            top_grasps: Liste von ValidatedGrasp (nur passed)
            pointclouds: Liste von (points, colors) pro Objekt
            window_name: Fenster-Titel
        """
        geometries = []
        
        # Objekt-Punktwolken
        rng = np.random.default_rng(seed=42)
        
        for obj_id, (points, colors) in enumerate(pointclouds):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Zufällige Farbe
            obj_color = rng.uniform(0.3, 1.0, size=3)
            colored = np.tile(obj_color, (len(points), 1))
            pcd.colors = o3d.utility.Vector3dVector(colored)
            
            geometries.append(pcd)
        
        # Top Grasps (nur grüne Pfeile)
        for i, grasp in enumerate(top_grasps):
            geom = self._create_grasp_geometry(grasp, rank=i+1)
            geometries.extend(geom)
        
        # Koordinatensystem
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)
        
        print(f"[GraspVisualizer] Öffne {window_name}...")
        print(f"[GraspVisualizer] Top {len(top_grasps)} Grasps")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1280,
            height=720
        )
    
    def visualize_heatmap(
        self,
        pointcloud: Tuple[np.ndarray, np.ndarray],
        grasp_scores: np.ndarray,
        window_name: str = "Grasp Quality Heatmap"
    ):
        """
        Zeigt Heatmap der Greifqualität auf der Oberfläche.
        
        Args:
            pointcloud: (points, colors) des Objekts
            grasp_scores: Score pro Punkt (0-1)
            window_name: Fenster-Titel
        """
        points, _ = pointcloud
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Farbkodierung: Rot (0) -> Gelb (0.5) -> Grün (1)
        colors = np.zeros((len(points), 3))
        
        for i, score in enumerate(grasp_scores):
            if score < 0.5:
                # Rot -> Gelb
                t = score * 2  # 0-1
                colors[i] = np.array([1.0, t, 0.0])
            else:
                # Gelb -> Grün
                t = (score - 0.5) * 2  # 0-1
                colors[i] = np.array([1.0 - t, 1.0, 0.0])
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print(f"[GraspVisualizer] Öffne {window_name}...")
        print(f"  Rot   = Niedriger Score (< 0.5)")
        print(f"  Gelb  = Mittlerer Score (~0.5)")
        print(f"  Grün  = Hoher Score (> 0.5)")
        
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=1280,
            height=720
        )
    
    def _create_grasp_geometry(
        self,
        grasp: ValidatedGrasp,
        rank: Optional[int] = None
    ) -> List[o3d.geometry.Geometry]:
        """
        Erstellt 3D-Geometrie für einen Greifkandidaten.
        
        Returns:
            Liste von Open3D Geometrien (Pfeil + Sphere)
        """
        geometries = []
        
        position = grasp.candidate.position
        normal = grasp.candidate.normal
        
        # Farbwahl basierend auf Validierung
        if grasp.passed:
            color = self.COLOR_PASSED
        elif grasp.verification.planarity_score < 0.3:
            color = self.COLOR_REJECTED
        else:
            color = self.COLOR_UNCERTAIN
        
        # 1. Pfeil für Normale
        arrow = self._create_arrow(
            start=position,
            direction=normal,
            length=self.arrow_length,
            color=color
        )
        geometries.append(arrow)
        
        # 2. Kugel am Greifpunkt
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.sphere_radius
        )
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
        
        # 3. Optional: Rank-Label (als kleine Kugel mit Offset)
        if rank is not None and rank <= 10:
            # Kleine Marker für Top-10
            label_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=self.sphere_radius * 0.5
            )
            # Offset seitlich
            label_pos = position + np.array([0.02, 0, 0])
            label_sphere.translate(label_pos)
            label_sphere.paint_uniform_color([1, 1, 1])  # Weiß
            geometries.append(label_sphere)
        
        return geometries
    
    def _create_arrow(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        length: float,
        color: np.ndarray
    ) -> o3d.geometry.TriangleMesh:
        """
        Erstellt einen 3D-Pfeil.
        
        Args:
            start: Startpunkt (3,)
            direction: Richtung (normalisiert) (3,)
            length: Länge des Pfeils
            color: RGB Farbe (3,)
            
        Returns:
            Arrow Mesh
        """
        # Normalisiere Richtung
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        
        # Erstelle Pfeil (zeigt in Z-Richtung)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=self.arrow_radius,
            cone_radius=self.arrow_radius * 2,
            cylinder_height=length * 0.7,
            cone_height=length * 0.3
        )
        
        # Rotiere von Z-Achse zu gewünschter Richtung
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            
            # Rodrigues' Rotation
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * angle
            )
            arrow.rotate(R, center=[0, 0, 0])
        
        # Verschiebe zu Startposition
        arrow.translate(start)
        
        # Färbe Pfeil
        arrow.paint_uniform_color(color)
        
        return arrow
    
    @staticmethod
    def create_comparison_view(
        validated_grasps: List[ValidatedGrasp],
        pointclouds: List[Tuple[np.ndarray, np.ndarray]],
        window_name: str = "All Grasps (Passed + Rejected)"
    ):
        """
        Statische Methode für schnelle Vergleichsvisualisierung.
        
        Zeigt alle Grasps (grün=bestanden, rot=abgelehnt) in einem Fenster.
        """
        visualizer = GraspVisualizer()
        visualizer.visualize_grasps(
            validated_grasps=validated_grasps,
            pointclouds=pointclouds,
            window_name=window_name
        )
    
    @staticmethod
    def create_top_view(
        top_grasps: List[ValidatedGrasp],
        pointclouds: List[Tuple[np.ndarray, np.ndarray]],
        window_name: str = "Top Validated Grasps"
    ):
        """
        Statische Methode für Top-N Visualisierung.
        
        Zeigt nur die besten validierten Grasps.
        """
        visualizer = GraspVisualizer()
        visualizer.visualize_top_grasps_only(
            top_grasps=top_grasps,
            pointclouds=pointclouds,
            window_name=window_name
        )

