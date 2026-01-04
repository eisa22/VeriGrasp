# tests/test_verification.py

import unittest
import numpy as np
import sys
import os

# Füge Parent-Directory zum Path hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GraspGeneration.suction_net import GraspCandidate
from Verification.geometric_verifier import GeometricVerifier, RejectionReason


class TestGeometricVerifier(unittest.TestCase):
    """Unit Tests für den GeometricVerifier."""
    
    def setUp(self):
        """Setup für jeden Test."""
        self.verifier = GeometricVerifier(
            planarity_threshold=0.02,
            normal_angle_threshold=15.0,
            min_sensor_quality=0.8,
            edge_distance_threshold=0.02,
            min_gripper_clearance=0.05
        )
    
    def test_workspace_bounds_check(self):
        """Test Workspace Bounds Validation."""
        # Erstelle Verifier mit Bounds
        verifier = GeometricVerifier(
            workspace_bounds=(
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 1.0])
            )
        )
        
        # Position innerhalb Bounds
        pos_inside = np.array([0.5, 0.5, 0.5])
        self.assertTrue(verifier._check_workspace_bounds(pos_inside))
        
        # Position außerhalb Bounds
        pos_outside = np.array([1.5, 0.5, 0.5])
        self.assertFalse(verifier._check_workspace_bounds(pos_outside))
    
    def test_planarity_check(self):
        """Test Planarity Check mit synthetischen Daten."""
        import open3d as o3d
        
        # Erstelle planare Oberfläche (XY-Ebene bei z=0)
        x = np.linspace(-0.1, 0.1, 20)
        y = np.linspace(-0.1, 0.1, 20)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        
        points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        # Test Position in der Mitte
        test_pos = np.array([0.0, 0.0, 0.0])
        
        planarity_score, is_planar = self.verifier._check_planarity(
            test_pos, pcd, tree
        )
        
        # Planare Fläche sollte hohen Score haben
        self.assertGreater(planarity_score, 0.9)
        self.assertTrue(is_planar)
    
    def test_normal_alignment(self):
        """Test Normal Alignment Check."""
        import open3d as o3d
        
        # Erstelle einfache Punktwolke
        points = np.random.randn(100, 3) * 0.1
        normals = np.tile([0, 0, 1], (100, 1))  # Alle Normalen zeigen nach oben
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        # Grasp-Normale zeigt auch nach oben
        test_pos = np.array([0.0, 0.0, 0.0])
        grasp_normal = np.array([0.0, 0.0, 1.0])
        
        angle, is_aligned = self.verifier._check_normal_alignment(
            test_pos, grasp_normal, points, normals, tree
        )
        
        # Sollte aligned sein (Winkel ~0°)
        self.assertLess(angle, 5.0)
        self.assertTrue(is_aligned)
        
        # Test mit falsch ausgerichteter Normale
        grasp_normal_wrong = np.array([1.0, 0.0, 0.0])  # Horizontal
        
        angle_wrong, is_aligned_wrong = self.verifier._check_normal_alignment(
            test_pos, grasp_normal_wrong, points, normals, tree
        )
        
        # Sollte nicht aligned sein
        self.assertGreater(angle_wrong, 45.0)
        self.assertFalse(is_aligned_wrong)
    
    def test_edge_proximity(self):
        """Test Edge Proximity Check."""
        import open3d as o3d
        
        # Erstelle Punktwolke mit klarem Rand
        # Zentrum: dichte Punkte, Rand: wenige Punkte
        center_points = np.random.randn(200, 3) * 0.05
        edge_point = np.array([[0.5, 0.5, 0.0]])
        
        points = np.vstack([center_points, edge_point])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        # Test Position im Zentrum (sicher)
        test_pos_center = np.array([0.0, 0.0, 0.0])
        
        dist_center, is_safe_center = self.verifier._check_edge_proximity(
            test_pos_center, pcd, tree
        )
        
        self.assertTrue(is_safe_center)
        
        # Test Position am Rand (unsicher)
        test_pos_edge = np.array([0.5, 0.5, 0.0])
        
        dist_edge, is_safe_edge = self.verifier._check_edge_proximity(
            test_pos_edge, pcd, tree
        )
        
        self.assertFalse(is_safe_edge)
    
    def test_full_verification_pipeline(self):
        """Test vollständige Verifikation mit synthetischen Kandidaten."""
        import open3d as o3d
        
        # Erstelle planare Oberfläche
        x = np.linspace(-0.1, 0.1, 30)
        y = np.linspace(-0.1, 0.1, 30)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        
        points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        colors = np.random.rand(len(points), 3)
        
        pointcloud = (points, colors)
        
        # Erstelle Kandidaten
        candidates = [
            # Guter Kandidat: zentral, nach oben gerichtet
            GraspCandidate(
                position=np.array([0.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                score=0.9,
                quality=0.9,
                object_id=0
            ),
            # Schlechter Kandidat: falsche Normale
            GraspCandidate(
                position=np.array([0.05, 0.05, 0.0]),
                normal=np.array([1.0, 0.0, 0.0]),  # Horizontal
                score=0.8,
                quality=0.7,
                object_id=0
            )
        ]
        
        # Verifiziere
        validated = self.verifier.verify_candidates(
            candidates=candidates,
            pointcloud=pointcloud,
            surface_normals=None,
            depth_map=None,
            camera_intrinsics=None
        )
        
        # Erwartung: 1 bestanden, 1 abgelehnt
        passed = [v for v in validated if v.passed]
        rejected = [v for v in validated if not v.passed]
        
        self.assertEqual(len(passed), 1)
        self.assertEqual(len(rejected), 1)
        
        # Der gute Kandidat sollte bestanden haben
        self.assertTrue(validated[0].passed)
        self.assertEqual(validated[0].verification.reason, RejectionReason.PASSED)


class TestGraspCandidate(unittest.TestCase):
    """Unit Tests für GraspCandidate."""
    
    def test_grasp_candidate_creation(self):
        """Test GraspCandidate Erstellung und Normalisierung."""
        candidate = GraspCandidate(
            position=[1.0, 2.0, 3.0],
            normal=[1.0, 1.0, 1.0],  # Nicht normalisiert
            score=0.8,
            quality=0.9,
            object_id=5
        )
        
        # Normale sollte automatisch normalisiert werden
        normal_norm = np.linalg.norm(candidate.normal)
        self.assertAlmostEqual(normal_norm, 1.0, places=5)
        
        # Position sollte float32 sein
        self.assertEqual(candidate.position.dtype, np.float32)
        
        # Object ID sollte korrekt sein
        self.assertEqual(candidate.object_id, 5)


if __name__ == '__main__':
    unittest.main()

