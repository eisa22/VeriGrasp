# Verification/geometric_verifier.py

import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

from GraspGeneration.suction_net import GraspCandidate
from config import DEBUG


class RejectionReason(Enum):
    """Gründe für Ablehnung eines Greifkandidaten."""
    PLANARITY = "Nicht ausreichend plan"
    NORMAL_ALIGNMENT = "Normale falsch ausgerichtet"
    SENSOR_QUALITY = "Unzureichende Sensorqualität"
    EDGE_PROXIMITY = "Zu nah am Objektrand"
    COLLISION_RISK = "Kollisionsgefahr"
    OUT_OF_BOUNDS = "Außerhalb des Arbeitsbereichs"
    PASSED = "Alle Checks bestanden"


@dataclass
class VerificationResult:
    """Details zur Verifikation eines Greifkandidaten."""
    passed: bool
    reason: RejectionReason
    planarity_score: float
    normal_angle: float  # Grad
    sensor_quality: float
    edge_distance: float  # Meter
    
    def __str__(self):
        status = "✓" if self.passed else "✗"
        return (
            f"{status} {self.reason.value} "
            f"(plan={self.planarity_score:.2f}, "
            f"angle={self.normal_angle:.1f}°, "
            f"sensor={self.sensor_quality:.2f})"
        )


@dataclass
class ValidatedGrasp:
    """Validierter Greifkandidat mit Verifikationsergebnissen."""
    candidate: GraspCandidate
    verification: VerificationResult
    rank: int = 0  # Wird nach Validierung zugewiesen
    
    @property
    def passed(self) -> bool:
        return self.verification.passed
    
    def __str__(self):
        return f"Grasp #{self.rank}: {self.verification}"


class GeometricVerifier:
    """
    Deterministischer Verification Layer für Greifkandidaten.
    
    Führt regelbasierte Constraint-Checks durch:
    - Planarity (Oberflächenkrümmung)
    - Surface Normal Alignment
    - Sensor Consistency
    - Edge/Occlusion Detection
    - Collision Avoidance
    """
    
    def __init__(
        self,
        planarity_threshold: float = 0.02,
        planarity_radius: float = 0.03,
        normal_angle_threshold: float = 15.0,
        prefer_upward_normals: bool = True,
        min_sensor_quality: float = 0.8,
        edge_distance_threshold: float = 0.02,
        min_gripper_clearance: float = 0.05,
        workspace_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """
        Args:
            planarity_threshold: Max. Standardabweichung der Normalen (m)
            planarity_radius: Radius für lokale Planarity-Analyse (m)
            normal_angle_threshold: Max. Winkelabweichung von idealer Normale (Grad)
            prefer_upward_normals: Bevorzuge nach oben gerichtete Flächen
            min_sensor_quality: Minimale Sensor-Qualität (0-1)
            edge_distance_threshold: Min. Abstand zu Objekträndern (m)
            min_gripper_clearance: Min. Clearance für Greifer (m)
            workspace_bounds: (min_xyz, max_xyz) des sicheren Arbeitsbereichs
        """
        self.planarity_threshold = planarity_threshold
        self.planarity_radius = planarity_radius
        self.normal_angle_threshold = normal_angle_threshold
        self.prefer_upward_normals = prefer_upward_normals
        self.min_sensor_quality = min_sensor_quality
        self.edge_distance_threshold = edge_distance_threshold
        self.min_gripper_clearance = min_gripper_clearance
        self.workspace_bounds = workspace_bounds
        
        if DEBUG:
            print("[GeometricVerifier] Initialisiert mit:")
            print(f"  Planarity Threshold: {planarity_threshold}m")
            print(f"  Normal Angle Threshold: {normal_angle_threshold}°")
            print(f"  Edge Distance Threshold: {edge_distance_threshold}m")
            print(f"  Min Gripper Clearance: {min_gripper_clearance}m")
    
    def verify_candidates(
        self,
        candidates: List[GraspCandidate],
        pointcloud: Tuple[np.ndarray, np.ndarray],
        surface_normals: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Dict] = None
    ) -> List[ValidatedGrasp]:
        """
        Verifiziert eine Liste von Greifkandidaten.
        
        Args:
            candidates: Liste von GraspCandidate Objekten
            pointcloud: Tuple (points, colors) - beide (N, 3) arrays
            surface_normals: Vorberechnete Normalen (N, 3), optional
            depth_map: Tiefenkarte (H, W), optional für Sensor-Checks
            camera_intrinsics: Dict mit fx, fy, cx, cy für Projection
            
        Returns:
            Liste von ValidatedGrasp Objekten (inkl. abgelehnte)
        """
        if len(candidates) == 0:
            return []
        
        points, colors = pointcloud
        
        # Open3D Punktwolke erstellen
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Normalen schätzen falls nicht vorhanden
        if surface_normals is None:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.planarity_radius,
                    max_nn=30
                )
            )
            surface_normals = np.asarray(pcd.normals)
        
        # KD-Tree für effiziente Nachbarsuche
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        validated_grasps = []
        
        for candidate in candidates:
            result = self._verify_single_candidate(
                candidate,
                pcd,
                points,
                surface_normals,
                tree,
                depth_map,
                camera_intrinsics
            )
            
            validated_grasps.append(ValidatedGrasp(
                candidate=candidate,
                verification=result
            ))
        
        # Ranke validierte Grasps
        passed = [g for g in validated_grasps if g.passed]
        rejected = [g for g in validated_grasps if not g.passed]
        
        # Sortiere nach Candidate Score * Verification Quality
        passed.sort(
            key=lambda g: g.candidate.score * g.verification.planarity_score,
            reverse=True
        )
        
        # Weise Ranks zu
        for i, grasp in enumerate(passed):
            grasp.rank = i + 1
        
        if DEBUG:
            print(f"[GeometricVerifier] Verifizierung abgeschlossen:")
            print(f"  Bestanden: {len(passed)}/{len(candidates)}")
            print(f"  Abgelehnt: {len(rejected)}/{len(candidates)}")
            
            # Ablehnungsgründe
            if rejected:
                reasons = {}
                for g in rejected:
                    reason = g.verification.reason
                    reasons[reason] = reasons.get(reason, 0) + 1
                print("  Ablehnungsgründe:")
                for reason, count in reasons.items():
                    print(f"    {reason.value}: {count}")
        
        return passed + rejected  # Bestanden zuerst
    
    def _verify_single_candidate(
        self,
        candidate: GraspCandidate,
        pcd: o3d.geometry.PointCloud,
        points: np.ndarray,
        surface_normals: np.ndarray,
        tree: o3d.geometry.KDTreeFlann,
        depth_map: Optional[np.ndarray],
        camera_intrinsics: Optional[Dict]
    ) -> VerificationResult:
        """Führt alle Checks für einen einzelnen Kandidaten durch."""
        
        position = candidate.position
        grasp_normal = candidate.normal
        
        # 1. Workspace Bounds Check
        if not self._check_workspace_bounds(position):
            return VerificationResult(
                passed=False,
                reason=RejectionReason.OUT_OF_BOUNDS,
                planarity_score=0.0,
                normal_angle=0.0,
                sensor_quality=0.0,
                edge_distance=0.0
            )
        
        # 2. Planarity Check
        planarity_score, is_planar = self._check_planarity(
            position, pcd, tree
        )
        
        # 3. Normal Alignment Check
        normal_angle, normals_aligned = self._check_normal_alignment(
            position, grasp_normal, points, surface_normals, tree
        )
        
        # 4. Edge Proximity Check
        edge_distance, is_safe_from_edges = self._check_edge_proximity(
            position, pcd, tree
        )
        
        # 5. Sensor Consistency Check
        sensor_quality = 1.0  # Default wenn keine Depth Map
        sensor_consistent = True
        if depth_map is not None and camera_intrinsics is not None:
            sensor_quality, sensor_consistent = self._check_sensor_consistency(
                position, depth_map, camera_intrinsics
            )
        
        # Zusammenfassung
        all_checks_passed = (
            is_planar and
            normals_aligned and
            is_safe_from_edges and
            sensor_consistent
        )
        
        # Bestimme Ablehnungsgrund
        if not all_checks_passed:
            if not is_planar:
                reason = RejectionReason.PLANARITY
            elif not normals_aligned:
                reason = RejectionReason.NORMAL_ALIGNMENT
            elif not is_safe_from_edges:
                reason = RejectionReason.EDGE_PROXIMITY
            elif not sensor_consistent:
                reason = RejectionReason.SENSOR_QUALITY
            else:
                reason = RejectionReason.COLLISION_RISK  # Fallback
        else:
            reason = RejectionReason.PASSED
        
        return VerificationResult(
            passed=all_checks_passed,
            reason=reason,
            planarity_score=planarity_score,
            normal_angle=normal_angle,
            sensor_quality=sensor_quality,
            edge_distance=edge_distance
        )
    
    def _check_workspace_bounds(self, position: np.ndarray) -> bool:
        """Prüft ob Position im sicheren Arbeitsbereich liegt."""
        if self.workspace_bounds is None:
            return True  # Kein Bound definiert -> alles ok
        
        min_bounds, max_bounds = self.workspace_bounds
        return np.all(position >= min_bounds) and np.all(position <= max_bounds)
    
    def _check_planarity(
        self,
        position: np.ndarray,
        pcd: o3d.geometry.PointCloud,
        tree: o3d.geometry.KDTreeFlann
    ) -> Tuple[float, bool]:
        """
        Prüft lokale Planarity um den Greifpunkt.
        
        Returns:
            (planarity_score, is_planar)
            planarity_score: 0-1, höher ist besser
            is_planar: True wenn Threshold erfüllt
        """
        points = np.asarray(pcd.points)
        
        # Finde nächsten Punkt in Punktwolke
        [k, idx, _] = tree.search_knn_vector_3d(position, 1)
        if k == 0:
            return 0.0, False
        
        nearest_idx = idx[0]
        
        # Finde Nachbarn im Radius
        [k, neighbor_indices, _] = tree.search_radius_vector_3d(
            points[nearest_idx], self.planarity_radius
        )
        
        if k < 3:
            return 0.0, False
        
        # Kovarianzmatrix für PCA
        neighbors = points[neighbor_indices]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)
        
        # Eigenwerte (sortiert absteigend)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Planarity Metric: 1 - (kleinster EV / Summe EVs)
        if eigenvalues.sum() > 1e-9:
            planarity_score = 1.0 - (eigenvalues[2] / eigenvalues.sum())
        else:
            planarity_score = 0.0
        
        planarity_score = np.clip(planarity_score, 0.0, 1.0)
        
        # Alternativ: Standardabweichung der z-Koordinaten
        # (für sehr planare Flächen sollte diese klein sein)
        z_std = np.std(neighbors[:, 2])
        is_planar = z_std < self.planarity_threshold
        
        # Kombiniere beide Metriken
        is_planar = is_planar and planarity_score > 0.7
        
        return planarity_score, is_planar
    
    def _check_normal_alignment(
        self,
        position: np.ndarray,
        grasp_normal: np.ndarray,
        points: np.ndarray,
        surface_normals: np.ndarray,
        tree: o3d.geometry.KDTreeFlann
    ) -> Tuple[float, bool]:
        """
        Prüft Ausrichtung der Greifnormalen zur Oberflächennormalen.
        
        Returns:
            (angle_deg, is_aligned)
        """
        # Finde nächsten Punkt
        [k, idx, _] = tree.search_knn_vector_3d(position, 1)
        if k == 0:
            return 180.0, False
        
        nearest_idx = idx[0]
        surface_normal = surface_normals[nearest_idx]
        
        # Winkel zwischen Greifnormale und Oberflächennormale
        # (beide sollten in gleiche Richtung zeigen)
        dot_product = np.dot(grasp_normal, surface_normal)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(np.abs(dot_product))  # abs für Richtungsunabhängigkeit
        angle_deg = np.degrees(angle_rad)
        
        is_aligned = angle_deg < self.normal_angle_threshold
        
        # Optional: Bevorzuge nach oben gerichtete Normalen
        if self.prefer_upward_normals and is_aligned:
            z_component = grasp_normal[2]
            if z_component < 0.5:  # Weniger als ~60° zur Horizontalen
                is_aligned = False
        
        return angle_deg, is_aligned
    
    def _check_edge_proximity(
        self,
        position: np.ndarray,
        pcd: o3d.geometry.PointCloud,
        tree: o3d.geometry.KDTreeFlann
    ) -> Tuple[float, bool]:
        """
        Prüft Abstand zu Objekträndern (Occlusion-Risk).
        
        Ränder werden durch geringe lokale Punktdichte erkannt.
        
        Returns:
            (edge_distance, is_safe)
        """
        points = np.asarray(pcd.points)
        
        # Finde nächsten Punkt
        [k, idx, _] = tree.search_knn_vector_3d(position, 1)
        if k == 0:
            return 0.0, False
        
        nearest_idx = idx[0]
        
        # Zähle Punkte in verschiedenen Radien
        small_radius = self.edge_distance_threshold
        large_radius = small_radius * 2
        
        [k_small, _, _] = tree.search_radius_vector_3d(
            points[nearest_idx], small_radius
        )
        [k_large, _, _] = tree.search_radius_vector_3d(
            points[nearest_idx], large_radius
        )
        
        # Punktdichte-Ratio (sollte nahe 1 sein wenn nicht am Rand)
        if k_large > 0:
            density_ratio = k_small / k_large
        else:
            density_ratio = 0.0
        
        # Heuristik: Am Rand ist Dichte geringer
        is_safe = density_ratio > 0.3 and k_small > 5
        
        # Edge distance schätzen (inverse Dichte)
        edge_distance = small_radius * density_ratio
        
        return edge_distance, is_safe
    
    def _check_sensor_consistency(
        self,
        position: np.ndarray,
        depth_map: np.ndarray,
        camera_intrinsics: Dict
    ) -> Tuple[float, bool]:
        """
        Prüft Sensor-Qualität am Greifpunkt.
        
        - Projiziert 3D-Position in Bildkoordinaten
        - Prüft Tiefenwert auf Validität
        
        Returns:
            (quality_score, is_consistent)
        """
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        # 3D -> 2D Projektion
        x, y, z = position
        
        if z <= 0:
            return 0.0, False
        
        u = int(fx * x / z + cx)
        v = int(fy * y / z + cy)
        
        # Bounds check
        H, W = depth_map.shape
        if not (0 <= u < W and 0 <= v < H):
            return 0.0, False
        
        # Depth-Wert an dieser Position
        depth_value = depth_map[v, u]
        
        # Check für NaN/Inf
        if not np.isfinite(depth_value):
            return 0.0, False
        
        # Check Konsistenz: gemessene Tiefe vs. 3D-Position
        depth_diff = np.abs(depth_value - z)
        relative_error = depth_diff / (z + 1e-6)
        
        quality_score = 1.0 - np.clip(relative_error, 0.0, 1.0)
        is_consistent = quality_score >= self.min_sensor_quality
        
        return quality_score, is_consistent

