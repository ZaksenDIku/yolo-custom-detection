"""
predict.py - kører den trænede model på billeder og gemmer billeder,
hvor forudsigelser og bounding boxes er tegnet ind (til screenshots).

Kør sådan (efter træning):
    source .venv/bin/activate
    python predict.py                      # bruger testbillederne i datasættet
    python predict.py --source sti/til/mappe_eller_billede
    python predict.py --conf 0.4           # højere sikkerhedstærskel = færre bokse

Billederne med bokse gemmes i runs/predict/billeder/.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# --- Stier ---
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS = BASE_DIR / "runs" / "train" / "weights" / "best.pt"  # trænede vægte
DEFAULT_SOURCE = BASE_DIR / "dataset_fixed" / "test" / "images"  # testbilleder fra datasættet

# --- Enhed ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Kør YOLO11 på billeder")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="billede eller mappe")
    parser.add_argument("--weights", default=str(WEIGHTS), help="sti til trænede vægte")
    parser.add_argument("--conf", type=float, default=0.25, help="sikkerhedstærskel (0-1)")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Fandt ingen vægte i {args.weights}. Kør train.py først.")
        return

    # Indlæs den trænede model.
    model = YOLO(args.weights)

    # Kør forudsigelser. save=True tegner bounding boxes ind og gemmer billederne.
    results = model.predict(
        source=args.source,
        conf=args.conf,
        device=DEVICE,
        save=True,
        project=str(BASE_DIR / "runs" / "predict"),
        name="billeder",
        exist_ok=True,
    )

    out_dir = BASE_DIR / "runs" / "predict" / "billeder"
    print(f"\nFærdig. Gemte {len(results)} billede(r) med bounding boxes i:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
