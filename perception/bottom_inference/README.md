# Bottom plane inference (Stage 2.5)

Infers each parcel's bottom height from lateral neighbours and pallet geometry when the top-down camera only sees the top surface.

## Entry point

```python
from perception.bottom_inference import infer_bottom_planes
from perception.configs.load import load_bottom_inference_config

config = load_bottom_inference_config()
candidates = infer_bottom_planes(candidates, scene_pcd, pallet_plane, config)
```

## Configuration

Loaded from [`../configs/bottom_inference.yaml`](../configs/bottom_inference.yaml). Code defaults in `perception/configs/load.py`:

| Key | Default | Meaning |
|-----|---------|---------|
| `lateral_radius_m` | 0.30 | Base lateral neighbour search radius |
| `min_neighbors` | 2 | Minimum neighbours before fallbacks |
| `height_tolerance` | 0.005 | Neighbour must be this much lower (m) |
| `tolerance_m` | 0.008 | Case A/B/C decision band (m) |
| `pallet_height_tolerance` | 0.030 | Layer-1 pallet proximity (m) |
| `edge_distance_m` | 0.05 | Boundary parcel margin (reserved) |
| `scene_pcd_stride` | 4 | Scene cloud subsampling in `main.py` |
| `obb.min_points` | 50 | Min points for OBB fit |
| `obb.max_aspect_ratio` | 20.0 | Reject degenerate OBB |

## Methods

- `measured` — visible minimum Z matches neighbours
- `from_neighbor` — augmented downward from neighbour tops
- `from_pallet` — isolated parcel on first layer
- `uncertain` — low-confidence estimate

## Tests

```bash
pytest tests/test_bottom_inference.py -q
```
