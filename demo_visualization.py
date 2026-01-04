#!/usr/bin/env python3
# demo_visualization.py

"""
Demo-Script für verschiedene Visualisierungsmodi.

Zeigt:
1. Alle Greifkandidaten (bestanden + abgelehnt)
2. Nur Top-N validierte Grasps
3. Optional: Heatmap der Greifqualität

Usage:
    python demo_visualization.py
    python demo_visualization.py --mode all
    python demo_visualization.py --mode top
"""

import sys
import argparse
import numpy as np

from GroundingSAM.grounding_sam import run_grounding_sam
from Sam3D.sam3d import SAM3D
from GraspGeneration.suction_net import SuctionNetWrapper
from Verification.geometric_verifier import GeometricVerifier
from Visualization.grasp_visualizer import GraspVisualizer
from path_utils import get_session_path, get_depth_path
from config import (
    SUCTIONNET_MODEL_PATH,
    SUCTIONNET_SCORE_THRESHOLD,
    MAX_GRASP_CANDIDATES_PER_OBJECT,
    MIN_POINTS_PER_GRASP,
    PLANARITY_THRESHOLD,
    PLANARITY_RADIUS,
    NORMAL_ANGLE_THRESHOLD,
    PREFER_UPWARD_NORMALS,
    MIN_SENSOR_QUALITY,
    EDGE_DISTANCE_THRESHOLD,
    MIN_GRIPPER_CLEARANCE,
    WORKSPACE_BOUNDS
)


def run_demo(mode='all'):
    """
    Führt Demo-Visualisierung aus.
    
    Args:
        mode: 'all' = alle Grasps, 'top' = nur Top-N, 'heatmap' = Qualitäts-Heatmap
    """
    print("="*60)
    print("VISUALIZATION DEMO")
    print("="*60)
    
    # 1. Perception
    print("\n[1/5] Running Perception...")
    boxes, masks, scores, labels = run_grounding_sam()
    
    if len(masks) == 0:
        print("❌ Keine Objekte erkannt!")
        return
    
    print(f"✓ {len(masks)} Objekte erkannt")
    
    # 2. 3D Reconstruction
    print("\n[2/5] 3D Reconstruction...")
    session_path = get_session_path()
    sam3d = SAM3D(session_path)
    pcs = sam3d.process(masks)
    
    print(f"✓ {len(pcs)} Punktwolken erstellt")
    
    # 3. Grasp Generation
    print("\n[3/5] Grasp Generation...")
    suction_net = SuctionNetWrapper(
        model_path=SUCTIONNET_MODEL_PATH,
        score_threshold=SUCTIONNET_SCORE_THRESHOLD,
        num_candidates=MAX_GRASP_CANDIDATES_PER_OBJECT,
        min_points_per_grasp=MIN_POINTS_PER_GRASP
    )
    
    all_candidates = []
    for obj_id, pc in enumerate(pcs):
        candidates = suction_net.predict_grasps(pc, object_id=obj_id)
        all_candidates.extend(candidates)
    
    print(f"✓ {len(all_candidates)} Kandidaten generiert")
    
    # 4. Verification
    print("\n[4/5] Verification...")
    verifier = GeometricVerifier(
        planarity_threshold=PLANARITY_THRESHOLD,
        planarity_radius=PLANARITY_RADIUS,
        normal_angle_threshold=NORMAL_ANGLE_THRESHOLD,
        prefer_upward_normals=PREFER_UPWARD_NORMALS,
        min_sensor_quality=MIN_SENSOR_QUALITY,
        edge_distance_threshold=EDGE_DISTANCE_THRESHOLD,
        min_gripper_clearance=MIN_GRIPPER_CLEARANCE,
        workspace_bounds=WORKSPACE_BOUNDS
    )
    
    # Lade Depth Map
    import os
    depth_path = get_depth_path()
    depth_map = np.load(depth_path) if os.path.exists(depth_path) else None
    
    camera_intrinsics = {
        'fx': sam3d.fx,
        'fy': sam3d.fy,
        'cx': sam3d.cx,
        'cy': sam3d.cy
    }
    
    all_validated = []
    for obj_id, pc in enumerate(pcs):
        obj_candidates = [c for c in all_candidates if c.object_id == obj_id]
        
        if len(obj_candidates) == 0:
            continue
        
        validated = verifier.verify_candidates(
            candidates=obj_candidates,
            pointcloud=pc,
            surface_normals=None,
            depth_map=depth_map,
            camera_intrinsics=camera_intrinsics
        )
        
        all_validated.extend(validated)
    
    passed_grasps = [g for g in all_validated if g.passed]
    rejected_grasps = [g for g in all_validated if not g.passed]
    
    print(f"✓ {len(passed_grasps)} bestanden, {len(rejected_grasps)} abgelehnt")
    
    # 5. Visualisierung
    print("\n[5/5] Visualization...")
    print(f"Mode: {mode}")
    
    visualizer = GraspVisualizer()
    
    if mode == 'all':
        # Zeige alle Grasps (grün + rot)
        visualizer.visualize_grasps(
            validated_grasps=all_validated,
            pointclouds=pcs,
            window_name="All Grasp Candidates (Green=Passed, Red=Rejected)"
        )
    
    elif mode == 'top':
        # Zeige nur Top-N bestandene
        if len(passed_grasps) == 0:
            print("❌ Keine validierten Grasps zum Anzeigen!")
            return
        
        # Top 5
        top_grasps = passed_grasps[:5]
        
        visualizer.visualize_top_grasps_only(
            top_grasps=top_grasps,
            pointclouds=pcs,
            window_name=f"Top {len(top_grasps)} Validated Grasps"
        )
    
    elif mode == 'heatmap':
        # Zeige Heatmap für erstes Objekt
        if len(pcs) == 0:
            print("❌ Keine Punktwolken zum Anzeigen!")
            return
        
        # Erstelle Score-Array (basierend auf Kandidaten)
        first_pc = pcs[0]
        points, _ = first_pc
        
        # Initialisiere Scores mit 0
        scores_map = np.zeros(len(points))
        
        # Setze Scores für Kandidaten
        first_obj_candidates = [c for c in all_candidates if c.object_id == 0]
        
        for candidate in first_obj_candidates:
            # Finde nächsten Punkt
            distances = np.linalg.norm(points - candidate.position, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Setze Score in Umgebung
            radius = 0.02  # 2cm
            in_range = distances < radius
            scores_map[in_range] = np.maximum(
                scores_map[in_range],
                candidate.score
            )
        
        visualizer.visualize_heatmap(
            pointcloud=first_pc,
            grasp_scores=scores_map,
            window_name="Grasp Quality Heatmap (Object 0)"
        )
    
    else:
        print(f"❌ Unbekannter Mode: {mode}")
        print("   Verfügbare Modi: 'all', 'top', 'heatmap'")
    
    print("\n" + "="*60)
    print("DEMO ABGESCHLOSSEN")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Visualisierungs-Demo für Vision-to-Grasp Pipeline'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['all', 'top', 'heatmap'],
        default='top',
        help='Visualisierungsmodus (default: top)'
    )
    
    args = parser.parse_args()
    
    try:
        run_demo(mode=args.mode)
    except KeyboardInterrupt:
        print("\n\n⚠ Demo abgebrochen durch Benutzer")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

