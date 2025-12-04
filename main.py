# main.py
from GroundingSAM.grounding_sam import run_grounding_sam  # liefert boxes, masks, scores, labels
from Sam3D.sam3d import SAM3D
from path_utils import get_session_path
from config import DEBUG


def main():
    # 1) GroundingDINO + SAM → 2D Masken
    # Erwartet: masks = Liste von 2D-Numpy-Arrays (H,W) mit 0/1 oder bool
    boxes, masks, scores, labels = run_grounding_sam()

    print(f"[MAIN] DINO+SAM lieferte {len(masks)} Masken.")

    # 2) SAM3D initialisieren
    session_path = get_session_path()
    sam3d = SAM3D(session_path)

    # 3) 2D Masken → 3D Punktwolken
    pcs = sam3d.process(masks)

    print(f"[MAIN] 3D Punktwolken erzeugt: {len(pcs)}")

    if not DEBUG:
        print("[MAIN] DEBUG=False, keine 3D-Visualisierung geöffnet.")


if __name__ == "__main__":
    main()
