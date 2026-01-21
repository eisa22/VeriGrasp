"""
Visualization/calibrate_viewpoints.py
Kalibrierungs-Tool zum Einstellen und Speichern von 3 Kamera-Viewpoints.

Anleitung:
1. Starte das Skript
2. Bewege die Kamera zur gewünschten Position (Rotation + Zoom)
3. Drücke 1, 2 oder 3 um den aktuellen Viewpoint zu speichern
4. Drücke S um alle Viewpoints in JSON zu speichern
5. Drücke Q oder schließe das Fenster um zu beenden
"""
import numpy as np
import open3d as o3d
import json
import os
from PIL import Image
import sys

# Pfad zum Projekt hinzufügen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SESSION_PATH


def load_pointcloud(session_path):
    """Lädt die RGBD-Punktwolke als Referenz."""
    rgb_path = os.path.join(session_path, "rgb", "rgb_0000.png")
    depth_path = os.path.join(session_path, "distance_to_image_plane", 
                               "distance_to_image_plane_0000.npy")
    
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        raise FileNotFoundError(f"RGB oder Depth nicht gefunden in {session_path}")
    
    rgb = np.array(Image.open(rgb_path))[:, :, :3]
    depth = np.load(depth_path)
    H, W = depth.shape
    
    # Kamera-Intrinsics
    fx = fy = 437.04
    cx, cy = W / 2, H / 2
    
    # Punktwolke erstellen
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    all_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # Transformation (Open3D Konvention)
    all_points[:, 1] *= -1
    all_points[:, 2] *= -1
    
    # RGB-Farben
    rgb_colors = rgb.reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
    
    return pcd


def get_viewpoint_json_path():
    """Gibt den Pfad zur Viewpoint-JSON-Datei zurück."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "viewpoints.json")


def save_viewpoints(viewpoints):
    """Speichert Viewpoints in JSON-Datei."""
    json_path = get_viewpoint_json_path()
    
    # Konvertiere numpy arrays zu Listen für JSON
    serializable = {}
    for key, vp in viewpoints.items():
        if vp is not None:
            serializable[key] = {
                "extrinsic": vp["extrinsic"].tolist(),
                "intrinsic": vp["intrinsic"].tolist(),
                "width": vp["width"],
                "height": vp["height"]
            }
    
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n[SAVE] Viewpoints gespeichert: {json_path}")
    return json_path


def load_viewpoints():
    """Lädt Viewpoints aus JSON-Datei falls vorhanden."""
    json_path = get_viewpoint_json_path()
    
    if not os.path.exists(json_path):
        return {"1": None, "2": None, "3": None}
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    viewpoints = {"1": None, "2": None, "3": None}
    for key in ["1", "2", "3"]:
        if key in data:
            viewpoints[key] = {
                "extrinsic": np.array(data[key]["extrinsic"]),
                "intrinsic": np.array(data[key]["intrinsic"]),
                "width": data[key]["width"],
                "height": data[key]["height"]
            }
    
    return viewpoints


def run_calibration():
    """Startet das Kalibrierungs-Tool."""
    print("\n" + "="*60)
    print("  VIEWPOINT KALIBRIERUNG")
    print("="*60)
    print("\nLade Punktwolke...")
    
    pcd = load_pointcloud(SESSION_PATH)
    viewpoints = load_viewpoints()
    
    # Status anzeigen
    print("\nGespeicherte Viewpoints:")
    for key in ["1", "2", "3"]:
        status = "✓ definiert" if viewpoints[key] else "✗ leer"
        print(f"  Viewpoint {key}: {status}")
    
    print("\n" + "-"*60)
    print("STEUERUNG:")
    print("  1, 2, 3  = Aktuellen View als Viewpoint 1/2/3 speichern")
    print("  S        = Alle Viewpoints in JSON exportieren")
    print("  L        = Viewpoint laden (1, 2 oder 3 danach drücken)")
    print("  Q        = Beenden")
    print("-"*60 + "\n")
    
    # Visualizer erstellen
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Viewpoint Kalibrierung", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # State für Load-Modus
    state = {"load_mode": False}
    
    def save_current_viewpoint(key):
        """Speichert den aktuellen Viewpoint."""
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        
        viewpoints[key] = {
            "extrinsic": np.array(cam.extrinsic),
            "intrinsic": np.array(cam.intrinsic.intrinsic_matrix),
            "width": cam.intrinsic.width,
            "height": cam.intrinsic.height
        }
        print(f"[OK] Viewpoint {key} gespeichert!")
    
    def load_viewpoint(key):
        """Lädt einen gespeicherten Viewpoint."""
        if viewpoints[key] is None:
            print(f"[FEHLER] Viewpoint {key} ist nicht definiert!")
            return
        
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        
        # Setze Kamera-Parameter
        cam.extrinsic = viewpoints[key]["extrinsic"]
        ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
        print(f"[OK] Viewpoint {key} geladen!")
    
    # Key Callbacks
    def on_key_1(vis):
        if state["load_mode"]:
            load_viewpoint("1")
            state["load_mode"] = False
        else:
            save_current_viewpoint("1")
        return False
    
    def on_key_2(vis):
        if state["load_mode"]:
            load_viewpoint("2")
            state["load_mode"] = False
        else:
            save_current_viewpoint("2")
        return False
    
    def on_key_3(vis):
        if state["load_mode"]:
            load_viewpoint("3")
            state["load_mode"] = False
        else:
            save_current_viewpoint("3")
        return False
    
    def on_key_s(vis):
        save_viewpoints(viewpoints)
        return False
    
    def on_key_l(vis):
        state["load_mode"] = True
        print("[INFO] Load-Modus aktiv - drücke 1, 2 oder 3 zum Laden")
        return False
    
    def on_key_q(vis):
        vis.destroy_window()
        return True
    
    # Registriere Callbacks
    vis.register_key_callback(ord("1"), on_key_1)
    vis.register_key_callback(ord("2"), on_key_2)
    vis.register_key_callback(ord("3"), on_key_3)
    vis.register_key_callback(ord("S"), on_key_s)
    vis.register_key_callback(ord("L"), on_key_l)
    vis.register_key_callback(ord("Q"), on_key_q)
    
    # Render-Optionen
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    vis.run()
    vis.destroy_window()
    
    print("\n[DONE] Kalibrierung beendet.\n")


if __name__ == "__main__":
    run_calibration()
