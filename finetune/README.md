# Fine-Tuning Pipeline

Scripts for fine-tuning a YOLOv8-nano model on aerial imagery to improve detection accuracy on the six GeoQuery classes.

## Why Fine-Tune?

Grounding DINO is a strong zero-shot baseline, but it has two weaknesses on aerial imagery: inference is slower (~2–3 seconds per pass), and small, densely-packed vehicles are frequently missed or duplicated. A YOLOv8 model trained specifically on aerial vehicle data runs in milliseconds and achieves significantly higher recall.

The `HybridDetector` in `models/yolo_detector.py` routes `vehicle` queries to the fine-tuned YOLO model and all other classes to Grounding DINO, combining the strengths of both.

---

## Quick Start (Google Colab)

### 1. Prepare a dataset

Any aerial object detection dataset in YOLO format works. Some publicly available options:

| Dataset | URL | Classes |
|---------|-----|---------|
| VisDrone | [github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) | vehicle (car, truck, bus) |
| iSAID | [captain-whu.github.io/iSAID](https://captain-whu.github.io/iSAID/) | building, vehicle, water |
| DOTA v2 | [captain-whu.github.io/DOTA](https://captain-whu.github.io/DOTA/dataset.html) | building, vehicle, road |
| LoveDA | [github.com/Junjue-Wang/LoveDA](https://github.com/Junjue-Wang/LoveDA) | building, road, water, vegetation |
| Roboflow Universe | [universe.roboflow.com](https://universe.roboflow.com) | Various, pre-formatted |

The VisDrone dataset was used to produce the `best.pt` weights included locally. Car, truck, and bus annotations were merged into a single `vehicle` class to match GeoQuery's class taxonomy.

### 2. Train

Upload `colab_train.py` to a Colab notebook (GPU runtime, T4 or better) and run it. Training takes roughly 2–3 hours on a T4.

Alternatively, to train locally with a GPU:

```bash
cd finetune
python train_local.py
```

### 3. Deploy the weights

After training, copy `best.pt` into the `finetune/` directory:

```
geoquery/
└── finetune/
    └── best.pt   ← place weights here
```

`config.py` checks for this file at startup and activates the `HybridDetector` automatically:

```python
USE_YOLO = os.path.exists("finetune/best.pt")
```

Restart the app. The sidebar will confirm which backend is active.

---

## Files

| File | Purpose |
|------|---------|
| `colab_train.py` | Self-contained training script for Google Colab |
| `train_local.py` | Local training script (GPU required) |
| `run_training.py` | Lightweight wrapper to kick off a local training run |
