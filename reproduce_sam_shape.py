
import numpy as np
import torch

# Mock logic from generate_sam_masks_from_roi
def verify_shape_logic():
    print("Verifying SAM shape logic...")
    
    # Simulate SAM output for 1 image, 2 boxes
    # Shape: (1, 2, 3, 100, 100) -> (Batch, NumBoxes, NumPrototypes, H, W)
    batch_masks = torch.zeros((1, 2, 3, 100, 100))
    batch_scores = torch.zeros((1, 2, 3))
    
    # Original code logic
    print(f"Original shape: masks={batch_masks.shape}, scores={batch_scores.shape}")
    
    # Code snippet lines 140-143
    if len(batch_masks.shape) == 4:
        print("Entering 4D logic (Correct path if squeezed)")
        batch_masks = batch_masks[:, 0, :, :]  # [num_boxes, H, W]
        batch_scores = batch_scores[:, 0]  # [num_boxes]
    else:
        print("SKIPPING 4D logic (Bug path)")
        
    # Conversion
    batch_masks_np = batch_masks.cpu().numpy().astype(np.uint8)
    batch_scores_np = batch_scores.cpu().numpy()
    
    print(f"Computed numpy shapes: masks={batch_masks_np.shape}, scores={batch_scores_np.shape}")
    
    all_masks = []
    all_scores = []
    min_area = 0
    
    # Loop
    for mask, score in zip(batch_masks_np, batch_scores_np):
        # If shape was (1, 2, 3, ...), iterate once (dim 0)
        # mask shape: (2, 3, 100, 100)
        # score shape: (2, 3)
        print(f"Inside loop - score type: {type(score)}, shape: {score.shape if hasattr(score, 'shape') else 'scalar'}")
        
        mask_area = np.sum(mask > 0)
        if mask_area >= min_area:
            all_masks.append(mask)
            all_scores.append(score)
            
    # Deduplicate check
    print(f"All scores len: {len(all_scores)}")
    if len(all_scores) > 0:
        print(f"First score element: {all_scores[0]}")
        
    # Simulate deduplicate sorting
    masks_np = all_masks # simplified
    scores_np = all_scores # simplified
    
    try:
        print("Attempting sort...")
        indices = sorted(range(len(masks_np)), key=lambda i: scores_np[i], reverse=True)
        print("Sort successful!")
    except ValueError as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    verify_shape_logic()
