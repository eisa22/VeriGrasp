"""Optional debug visualisation for bottom-plane inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perception.candidate import CandidateOut


def visualize_bottom_inference(
    candidates: list[CandidateOut],
    plane: tuple[float, float, float, float],
) -> None:
    """Draw OBB wireframes and neighbour links (requires Open3D GUI)."""
    import open3d as o3d

    geoms = []
    for c in candidates:
        if c.bottom is None:
            continue
        obb = c.bottom.parcel_obb
        corners = o3d.utility.Vector3dVector(obb["corners_3d"])
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
        ls = o3d.geometry.LineSet(corners, o3d.utility.Vector2iVector(lines))
        ls.paint_uniform_color([0.2, 0.8, 0.2])
        geoms.append(ls)

    if geoms:
        o3d.visualization.draw_geometries(geoms, window_name="Bottom inference debug")
