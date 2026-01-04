# main.py
from GroundingSAM.grounding_sam import run_grounding_sam
from Sam3D.sam3d import SAM3D
from path_utils import get_all_session_paths
from config import DEBUG
import os


def main():
    # Alle Sessions holen
    all_sessions = get_all_session_paths()
    print(f"[MAIN] Gefundene Sessions: {len(all_sessions)}")
    
    for session_path in all_sessions:
        session_name = os.path.basename(session_path)
        print(f"\n{'='*60}")
        print(f"[MAIN] Verarbeite Session: {session_name}")
        print(f"{'='*60}")
        
        # 1) GroundingDINO + SAM → 2D Masken
        # Diese Funktion zeigt automatisch die 2D-Visualisierung wenn DEBUG=True
        boxes, masks, scores, labels = run_grounding_sam(session_path)
        
        print(f"[MAIN] {session_name}: DINO+SAM lieferte {len(masks)} Masken.")
        
        if len(masks) == 0:
            print(f"[MAIN] {session_name}: Keine Masken gefunden, überspringe Session.")
            continue
        
        # 2) SAM3D initialisieren
        sam3d = SAM3D(session_path)
        
        # 3) 2D Masken → 3D Punktwolken
        # Diese Funktion zeigt automatisch die 3D-Visualisierung wenn DEBUG=True
        pcs = sam3d.process(masks)
        
        print(f"[MAIN] {session_name}: 3D Punktwolken erzeugt: {len(pcs)}")
    
    print(f"\n{'='*60}")
    print(f"[MAIN] Alle {len(all_sessions)} Sessions verarbeitet!")
    print(f"{'='*60}")
    
    if not DEBUG:
        print("[MAIN] DEBUG=False, keine Visualisierungen wurden angezeigt.")


if __name__ == "__main__":
    main()
