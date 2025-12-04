# main.py

from GroundingSAM.grounding_sam import run_grounding_sam
from Sam3D.sam3d import SAM3D

def main():
    # Schritt 1 – SAM 2D
    boxes, masks, scores, labels = run_grounding_sam()

    if len(masks) == 0:
        print("Keine Objekte gefunden.")
        return

    # Schritt 2 – SAM 3D
    sam3d = SAM3D()
    pointclouds = sam3d.process(masks)

    print("3D Punktwolken erzeugt:", len(pointclouds))

    # Beispiel: Zugriff auf erste Punktwolke
    # pc0 = pointclouds[0]

if __name__ == "__main__":
    main()
