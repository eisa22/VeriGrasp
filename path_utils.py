# path_utils.py
import os
from config import BASE_PATH

def get_rgb_path():
    return os.path.join(BASE_PATH, "rgb", "rgb_0000.png")

def get_pointcloud_path():
    return os.path.join(BASE_PATH, "pointcloud", "pointcloud_0000.npy")
