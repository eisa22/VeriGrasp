
import numpy as np
import open3d as o3d
from unittest.mock import MagicMock, patch
import sys
import os

# Put VisionPipeline on path
sys.path.append("/home/samuel/Thesis/VisionPipeline")

from Visualization.visualizer import visualize_3d

def test_visualize_3d():
    print("Testing visualize_3d...")
    
    # Mock data
    H, W = 100, 100
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.ones((H, W), dtype=np.float32) * 2.0 
    # Add noise to avoid planar degeneracy for OBB
    depth += np.random.rand(H, W) * 0.1
    
    # Create 2 masks
    mask1 = np.zeros((H, W), dtype=np.uint8)
    mask1[20:40, 20:40] = 1
    
    mask2 = np.zeros((H, W), dtype=np.uint8)
    mask2[60:80, 60:80] = 1
    
    masks = [mask1, mask2]
    labels = ["box1", "box2"]
    
    # Mock file loading
    with patch("Visualization.visualizer.np.load", return_value=depth), \
         patch("Visualization.visualizer.Image.open") as mock_img_open, \
         patch("Visualization.visualizer.o3d.visualization.draw_geometries") as mock_draw:
        
        # Mock Image.open result
        mock_img_open.return_value = MagicMock()
        mock_img_open.return_value.__array__ = lambda *args, **kwargs: rgb
        # Handle conversion which might be called
        mock_img_open.return_value.convert.return_value = mock_img_open.return_value
        
        # Call function
        visualize_3d("dummy_path", masks, labels)
        
        print("Function called successfully.")
        
        # Verify draw_geometries called
        assert mock_draw.called
        geoms = mock_draw.call_args[0][0]
        print(f"Number of geometries passed to draw: {len(geoms)}")
        
        # We expect: 
        # 1 full cloud
        # 2 segment clouds
        # 2 bounding boxes
        # Total = 5
        print(f"Geometries found: {len(geoms)}")
        
        # Check types
        pcds = [g for g in geoms if isinstance(g, o3d.geometry.PointCloud)]
        obbs = [g for g in geoms if isinstance(g, o3d.geometry.OrientedBoundingBox)]
        
        print(f"PointClouds: {len(pcds)}")
        print(f"OBBs: {len(obbs)}")
        
        if len(obbs) == 2:
            print("SUCCESS: 2 OBBs created as expected.")
        else:
            print(f"FAILURE: Expected 2 OBBs, found {len(obbs)}")
            sys.exit(1)

if __name__ == "__main__":
    test_visualize_3d()
