# YOLO11 på custom datasæt (TTCA)

Et lille Python-projekt der træner en **YOLO11**-model til at detektere køretøjer
(car, truck, bus, van) på billeder. Projektet er lavet til en obligatorisk
Machine Learning-aflevering.

## Hvad projektet indeholder

| Fil | Formål |
|---|---|
| `train.py` | Træner modellen på datasættet og gemmer vægte + plots. |
| `evaluate.py` | Beregner Precision, Recall og mAP50 og sammenligner med kravene. |
| `predict.py` | Kører modellen på billeder og gemmer billeder med bounding boxes. |
| `requirements.txt` | Python-afhængigheder. |
| `dataset/` | Datasættet (hentes separat - ligger ikke på GitHub). |

## Datasæt

- **Navn:** TTCA (version 2), 1062 billeder i alt.
- **Kilde:** <https://universe.roboflow.com/kakus-workspace/ttca>
- **Licens:** CC BY 4.0 (fri brug med kildeangivelse).
- **Klasser (4):** `bus`, `car`, `truck`, `van`.
- **Opdeling:** 942 træningsbilleder, 79 valideringsbilleder, 41 testbilleder.
- **Format:** YOLO (én `.txt`-fil pr. billede + `data.yaml`).

Datasættet downloades fra Roboflow som en ZIP og pakkes ud i `dataset/`-mappen.
Det er **ikke** inkluderet i git (se `.gitignore`).

## Krav til miljø

- macOS med Apple Silicon (testet på Apple M2).
- Python 3.10 eller nyere.

## Opsætning

Opret et virtuelt miljø og installér afhængighederne:

```bash
# Med standard-værktøjer:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternativt med `uv` (hurtigere):

```bash
uv venv .venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

Aktiver altid miljøet (`source .venv/bin/activate`) før du kører scripts.

## Kør projektet

### 1. Træn modellen

```bash
python train.py
```

- Starter fra fortrænede vægte `yolo11n.pt` (hentes automatisk første gang).
- Bruger Mac'ens GPU (MPS), hvis den er tilgængelig, ellers CPU.
- Standard: 20 epochs, billedstørrelse 512.
- Tiden afhænger af maskinen - på en M2 tager 50 epochs typisk et stykke tid
  (cirka en halv til en hel time). Se afsnittet "Ressourceforbrug" nedenfor.

Resultater gemmes i `runs/train/`, bl.a.:

- `runs/train/weights/best.pt` - de bedste vægte.
- `runs/train/results.csv` - målinger pr. epoch.
- `runs/train/results.png` - plot af træning og validering.
- `runs/train/confusion_matrix.png` - forvekslingsmatrix.

### 2. Evaluér modellen

```bash
python evaluate.py
```

Udskriver Precision (P), Recall (R) og mAP50 og markerer om hvert krav er opfyldt.

**Lærerens minimumskrav:** P > 0,5, R > 0,3, mAP50 > 0,2.

### 3. Test modellen på billeder

```bash
python predict.py                       # testbilleder fra datasættet
python predict.py --source mit_billede.jpg
python predict.py --source min_mappe/
```

Billederne med bounding boxes gemmes i `runs/predict/billeder/`.
Her kan du tage screenshots til itslearning.

## Ressourceforbrug

Træning bruger meget CPU/GPU og hukommelse. På en Mac med 8 GB RAM er
`batch=16` og `imgsz=640` et fornuftigt udgangspunkt. Hvis maskinen løber tør
for hukommelse, kan du sænke `BATCH` i `train.py` (fx til 8).

## Kildeangivelse

Datasættet er udgivet under CC BY 4.0 af brugeren "kakus Workspace" på Roboflow
Universe. Ved brug skal kilden angives:

> TTCA Dataset, kakus Workspace, Roboflow Universe,
> https://universe.roboflow.com/kakus-workspace/ttca
