"""
train.py - træner en YOLO11-model på TTCA-datasættet (biltedetektion).

Kør sådan:
    source .venv/bin/activate
    python train.py

Modellen starter fra fortrænede vægte (yolo11n.pt) og finjusteres på vores data.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

# --- Stier og indstillinger (ændr her, hvis du vil eksperimentere) ---
BASE_DIR = Path(__file__).resolve().parent  # mappen hvor dette script ligger
DATA_YAML = BASE_DIR / "dataset_fixed" / "data.yaml"  # rettede labels (polygoner -> bbox)
MODEL = "yolo11n.pt"  # fortrænet YOLO11 nano (lille og hurtig). Hentes automatisk første gang
EPOCHS = 20  # antal gennemløb af hele træningssættet
IMGSZ = 512  # billedstørrelse (mindre = hurtigere træning på denne Mac)
BATCH = 16  # antal billeder pr. træningsskridt (passer til 8 GB RAM)
WORKERS = 4  # antal processer til at indlæse billeder
PROJECT = BASE_DIR / "runs"  # mappe hvor resultater gemmes
NAME = "train"  # navn på denne træningskørsel

# Brug Mac'ens GPU (MPS) hvis den findes, ellers CPU.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print(f"Træner på enhed: {DEVICE}")
    print(f"Datasæt: {DATA_YAML}")

    # Indlæs den fortrænede model (arkitekturen + vægte fra COCO).
    model = YOLO(MODEL)

    # Start træningen. Ultralytics gemmer vægte, plots og logfiler under runs/.
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,  # overskriv/opdater samme mappe i stedet for at lave en ny
    )

    # Efter træning ligger de vigtigste målinger i results.box.
    print("\n=== Træning færdig ===")
    print(f"Precision (P): {results.box.mp:.3f}")
    print(f"Recall (R):    {results.box.mr:.3f}")
    print(f"mAP50:         {results.box.map50:.3f}")
    print(f"mAP50-95:      {results.box.map:.3f}")
    print(f"Bedste vægte gemt i: {PROJECT / NAME / 'weights' / 'best.pt'}")
    print("\nKør nu evaluate.py for den officielle evaluering på valideringsdata.")


if __name__ == "__main__":
    main()
