"""SAM-Grid adaptive routing (mocked, no GPU)."""

from unittest.mock import MagicMock, patch

import numpy as np

from perception.config_loader import SamConfig
from perception.sam_grid import _box_area_ratio, boxes_to_sam_masks_adaptive


def test_box_area_ratio():
    assert _box_area_ratio([0, 0, 50, 50], 100, 100) == 0.25


@patch("GroundingSAM.sam_grid_generator.generate_sam_masks_from_roi")
@patch("perception.sam_hf.boxes_to_sam_masks")
@patch("perception.sam_hf.get_sam")
def test_adaptive_uses_grid_for_large_roi(mock_get_sam, mock_box_sam, mock_grid_roi):
    mock_get_sam.return_value = (MagicMock(), MagicMock(), "id")
    mock_grid_roi.return_value = [np.ones((64, 64), dtype=np.uint8)]
    mock_box_sam.return_value = [np.ones((32, 32), dtype=np.uint8)]

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = SamConfig(grid_on_roi_area_frac=0.04)

    large = [0.0, 0.0, 80.0, 80.0]
    small = [0.0, 0.0, 10.0, 10.0]

    masks, scores, labels, boxes = boxes_to_sam_masks_adaptive(
        rgb,
        [large, small],
        ["a", "b"],
        [0.9, 0.8],
        cfg,
        sam_model=MagicMock(),
        sam_processor=MagicMock(),
    )

    mock_grid_roi.assert_called_once()
    mock_box_sam.assert_called_once()
    assert len(masks) == 2
    assert len(labels) == 2
