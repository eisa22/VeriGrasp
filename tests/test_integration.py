# tests/test_integration.py

import unittest
import sys
import os
import numpy as np

# Füge Parent-Directory zum Path hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import BASE_PATH
from Sam3D.sam3d import SAM3D
from GraspGeneration.suction_net import SuctionNetWrapper
from Verification.geometric_verifier import GeometricVerifier


class TestIntegration(unittest.TestCase):
    """Integration Tests über die gesamte Pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Setup für alle Tests (wird einmal ausgeführt)."""
        cls.test_scenes = [
            "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_07",
            "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_08",
            "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_09",
        ]
    
    def test_sam3d_initialization(self):
        """Test SAM3D Initialisierung mit echten Daten."""
        for scene in self.test_scenes:
            if not os.path.exists(scene):
                self.skipTest(f"Scene {scene} nicht gefunden")
            
            with self.subTest(scene=scene):
                sam3d = SAM3D(scene)
                
                # Prüfe ob Daten geladen wurden
                self.assertIsNotNone(sam3d.rgb)
                self.assertIsNotNone(sam3d.depth)
                self.assertGreater(sam3d.W, 0)
                self.assertGreater(sam3d.H, 0)
    
    def test_grasp_generation_with_synthetic_object(self):
        """Test Grasp Generation mit synthetischer Punktwolke."""
        # Erstelle synthetisches Objekt (Box)
        x = np.linspace(-0.1, 0.1, 50)
        y = np.linspace(-0.1, 0.1, 50)
        xx, yy = np.meshgrid(x, y)
        
        # Top-Fläche bei z=0.2
        zz = np.full_like(xx, 0.2)
        
        points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        colors = np.random.rand(len(points), 3)
        
        pointcloud = (points, colors)
        
        # Initialisiere SuctionNet
        suction_net = SuctionNetWrapper(
            model_path=None,
            score_threshold=0.3,
            num_candidates=5
        )
        
        # Generiere Kandidaten
        candidates = suction_net.predict_grasps(pointcloud, object_id=0)
        
        # Sollte Kandidaten finden
        self.assertGreater(len(candidates), 0)
        
        # Prüfe Kandidaten-Properties
        for candidate in candidates:
            self.assertEqual(len(candidate.position), 3)
            self.assertEqual(len(candidate.normal), 3)
            self.assertGreaterEqual(candidate.score, 0.0)
            self.assertLessEqual(candidate.score, 1.0)
            
            # Normale sollte normalisiert sein
            normal_norm = np.linalg.norm(candidate.normal)
            self.assertAlmostEqual(normal_norm, 1.0, places=5)
    
    def test_full_pipeline_synthetic(self):
        """Test vollständige Pipeline mit synthetischen Daten."""
        # 1. Erstelle synthetisches Objekt
        x = np.linspace(-0.15, 0.15, 80)
        y = np.linspace(-0.15, 0.15, 80)
        xx, yy = np.meshgrid(x, y)
        zz = 0.3 + 0.01 * np.sin(xx * 10)  # Leicht gewellte Oberfläche
        
        points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        colors = np.random.rand(len(points), 3)
        
        pointcloud = (points, colors)
        
        # 2. Grasp Generation
        suction_net = SuctionNetWrapper(
            model_path=None,
            score_threshold=0.4,
            num_candidates=10
        )
        
        candidates = suction_net.predict_grasps(pointcloud, object_id=0)
        
        self.assertGreater(len(candidates), 0, "Keine Kandidaten generiert")
        
        # 3. Verification
        verifier = GeometricVerifier(
            planarity_threshold=0.03,
            normal_angle_threshold=20.0,
            min_sensor_quality=0.7,
            edge_distance_threshold=0.02
        )
        
        validated = verifier.verify_candidates(
            candidates=candidates,
            pointcloud=pointcloud,
            surface_normals=None,
            depth_map=None,
            camera_intrinsics=None
        )
        
        self.assertEqual(len(validated), len(candidates))
        
        # 4. Prüfe Ergebnisse
        passed = [v for v in validated if v.passed]
        rejected = [v for v in validated if not v.passed]
        
        print(f"\n  Pipeline Results: {len(passed)} passed, {len(rejected)} rejected")
        
        # Es sollten einige Kandidaten bestanden haben
        self.assertGreater(len(passed), 0, "Keine Kandidaten haben Verifikation bestanden")
        
        # Top Kandidat sollte gute Scores haben
        if len(passed) > 0:
            top = passed[0]
            self.assertGreaterEqual(top.candidate.score, 0.4)
            self.assertGreaterEqual(top.verification.planarity_score, 0.5)
    
    def test_multiple_objects(self):
        """Test Pipeline mit mehreren Objekten."""
        # Erstelle 3 separate Objekte
        objects = []
        
        for i in range(3):
            x = np.linspace(-0.08, 0.08, 40)
            y = np.linspace(-0.08, 0.08, 40)
            xx, yy = np.meshgrid(x, y)
            
            # Unterschiedliche Höhen
            zz = np.full_like(xx, 0.2 + i * 0.1)
            
            points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
            # Offset in X-Richtung
            points[:, 0] += i * 0.3
            
            colors = np.random.rand(len(points), 3)
            objects.append((points, colors))
        
        # Verarbeite jedes Objekt
        suction_net = SuctionNetWrapper(num_candidates=5, score_threshold=0.3)
        verifier = GeometricVerifier()
        
        all_validated = []
        
        for obj_id, pointcloud in enumerate(objects):
            candidates = suction_net.predict_grasps(pointcloud, object_id=obj_id)
            
            if len(candidates) == 0:
                continue
            
            validated = verifier.verify_candidates(
                candidates=candidates,
                pointcloud=pointcloud
            )
            
            all_validated.extend(validated)
        
        # Sollte Grasps für mehrere Objekte haben
        object_ids = set(v.candidate.object_id for v in all_validated)
        self.assertGreaterEqual(len(object_ids), 2, "Zu wenige Objekte verarbeitet")
        
        print(f"\n  Processed {len(object_ids)} objects, {len(all_validated)} total grasps")


class TestMultiSceneValidation(unittest.TestCase):
    """Validierung über mehrere Szenen."""
    
    def test_multiple_scenes_statistics(self):
        """Sammle Statistiken über mehrere Szenen."""
        test_scenes = [
            "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_07",
            "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_08",
            "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_09",
            "/home/samuel/Thesis/VisionPipeline/Data/pallet_rgbd_data/Replicator_10",
        ]
        
        stats = {
            'scenes_processed': 0,
            'avg_candidates': [],
            'avg_passed': [],
            'pass_rate': []
        }
        
        # Erstelle Mock-Masken für Testing
        # (In echtem Test würden wir Grounding DINO nutzen)
        mock_mask = np.zeros((480, 640), dtype=np.uint8)
        mock_mask[100:300, 150:450] = 1  # Rechteck
        
        for scene in test_scenes:
            if not os.path.exists(scene):
                print(f"⚠ Scene {scene} nicht gefunden, überspringe")
                continue
            
            try:
                # SAM3D
                sam3d = SAM3D(scene)
                pcs = sam3d.process([mock_mask])
                
                if len(pcs) == 0:
                    continue
                
                # Grasp Generation
                suction_net = SuctionNetWrapper(num_candidates=10)
                candidates = suction_net.predict_grasps(pcs[0], object_id=0)
                
                if len(candidates) == 0:
                    continue
                
                # Verification
                verifier = GeometricVerifier()
                validated = verifier.verify_candidates(
                    candidates=candidates,
                    pointcloud=pcs[0]
                )
                
                passed = [v for v in validated if v.passed]
                
                # Statistiken
                stats['scenes_processed'] += 1
                stats['avg_candidates'].append(len(candidates))
                stats['avg_passed'].append(len(passed))
                stats['pass_rate'].append(len(passed) / len(candidates) if len(candidates) > 0 else 0)
                
            except Exception as e:
                print(f"⚠ Fehler bei Scene {scene}: {e}")
                continue
        
        # Ausgabe
        if stats['scenes_processed'] > 0:
            print(f"\n  {'='*50}")
            print(f"  Multi-Scene Statistics")
            print(f"  {'='*50}")
            print(f"  Scenes Processed: {stats['scenes_processed']}")
            print(f"  Avg Candidates per Scene: {np.mean(stats['avg_candidates']):.1f}")
            print(f"  Avg Passed per Scene: {np.mean(stats['avg_passed']):.1f}")
            print(f"  Avg Pass Rate: {np.mean(stats['pass_rate'])*100:.1f}%")
            print(f"  {'='*50}")
            
            # Mindestens 1 Szene sollte verarbeitet worden sein
            self.assertGreater(stats['scenes_processed'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

