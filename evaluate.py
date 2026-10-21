"""
evaluate.py - evaluerer den trænede model på valideringsdata.

Kør sådan (efter træning):
    source .venv/bin/activate
    python evaluate.py

Scriptet udskriver Precision (P), Recall (R) og mAP50 og sammenligner
dem med lærerens minimumskrav.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

# --- Stier ---
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset_fixed" / "data.yaml"
WEIGHTS = BASE_DIR / "runs" / "train" / "weights" / "best.pt"  # trænede vægte

# --- Enhed ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Lærerens minimumskrav ---
KRAV_P = 0.5
KRAV_R = 0.3
KRAV_MAP50 = 0.2


def main():
    if not WEIGHTS.exists():
        print(f"Fandt ingen vægte i {WEIGHTS}. Kør train.py først.")
        return

    # Indlæs de trænede vægte og kør validering på val-sættet.
    model = YOLO(str(WEIGHTS))
    metrics = model.val(data=str(DATA_YAML), split="val", device=DEVICE)

    # Hent tallene ud af resultatet.
    precision = metrics.box.mp  # gennemsnitlig Precision
    recall = metrics.box.mr  # gennemsnitlig Recall
    map50 = metrics.box.map50  # mAP ved IoU-grænse 0.50
    map5095 = metrics.box.map  # mAP over IoU 0.50-0.95

    print("\n=== Evaluering på valideringsdata ===")
    print(f"Precision (P): {precision:.3f}   (krav > {KRAV_P})")
    print(f"Recall (R):    {recall:.3f}   (krav > {KRAV_R})")
    print(f"mAP50:         {map50:.3f}   (krav > {KRAV_MAP50})")
    print(f"mAP50-95:      {map5095:.3f}")

    print("\n=== Sammenligning med krav ===")
    print("Precision:", "OPFYLDT" if precision > KRAV_P else "IKKE opfyldt")
    print("Recall:   ", "OPFYLDT" if recall > KRAV_R else "IKKE opfyldt")
    print("mAP50:    ", "OPFYLDT" if map50 > KRAV_MAP50 else "IKKE opfyldt")


if __name__ == "__main__":
    main()
