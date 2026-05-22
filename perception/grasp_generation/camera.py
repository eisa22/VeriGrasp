"""Pinhole camera helpers for suction grasp generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraInfo:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float = 1.0


def camera_from_shape(
    height: int,
    width: int,
    fx: float = 437.04,
    fy: float = 437.04,
) -> CameraInfo:
    return CameraInfo(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=width / 2.0,
        cy=height / 2.0,
        scale=1.0,
    )


def depth_to_point_cloud(depth: np.ndarray, camera: CameraInfo, organized: bool = True) -> np.ndarray:
    """Back-project depth (metres) to 3D points in camera frame."""
    assert depth.shape[0] == camera.height and depth.shape[1] == camera.width
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth.astype(np.float32)
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape(-1, 3)
    return cloud
