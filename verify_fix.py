
import torch
import numpy as np
from unittest.mock import MagicMock
from GroundingSAM.sam_grid_generator import generate_sam_masks_from_roi

def test_fix():
    print("Testing fix with Mock objects...")
    
    # Mock Model and Processor
    sam_model = MagicMock()
    sam_processor = MagicMock()
    
    # Mock Output
    # Shape: (1, 1, 3, 50, 50) -> Batch=1, NumBoxes=1, Prototypes=3, H=50, W=50
    mock_pred_masks = torch.rand(1, 1, 3, 50, 50)
    mock_iou_scores = torch.rand(1, 1, 3)
    
    sam_outputs = MagicMock()
    sam_outputs.pred_masks = mock_pred_masks
    sam_outputs.iou_scores = mock_iou_scores
    
    sam_model.return_value = sam_outputs
    
    # Mock device
    sam_model.parameters.side_effect = lambda: iter([torch.tensor([0])])
    
    # Mock inputs
    image = MagicMock()
    roi_box = [0, 0, 100, 100]
    
    # Run function
    try:
        masks = generate_sam_masks_from_roi(
            sam_model, 
            sam_processor, 
            image, 
            roi_box, 
            grid_size=1, # Minimal grid to trigger 1 box
            min_area=0
        )
        print(f"Success! Returned {len(masks)} masks.")
    except Exception as e:
        print(f"FAILED with error: {e}")
        raise e

if __name__ == "__main__":
    test_fix()
