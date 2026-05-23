"""Gemeinsame Hilfen für binäre Instanzmasken."""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage


def ensure_mask_hw(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Skaliert Maske auf (H, W); SAM-Rohausgaben sind oft 256×256."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape == (height, width):
        return binary
    resized = PILImage.fromarray(binary).resize((width, height), PILImage.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)
