# GeoQuery Fine-Tuning Pipeline

Fine-tune YOLOv8 on aerial imagery for dramatically better detection accuracy
on the 6 GeoQuery target classes.

## Why Fine-Tune?

| Approach | Pros | Cons |
|----------|------|------|
| **Grounding DINO (current)** | Zero-shot, no training needed | Slower, less accurate on aerial imagery |
| **YOLOv8 Fine-tuned** | Much faster, much more accurate | Requires training data + GPU time |

The PS explicitly rewards fine-tuning in the **innovation scoring** component.

## Quick Start (Google Colab)

### Step 1: Get a Dataset

**Recommended datasets** (all free, public):

| Dataset | Download | Best For |
|---------|----------|----------|
| **iSAID** | [captain-whu.github.io/iSAID](https://captain-whu.github.io/iSAID/) | building, vehicle, water |
| **DOTA v2** | [captain-whu.github.io/DOTA](https://captain-whu.github.io/DOTA/dataset.html) | building, vehicle, road |
| **LoveDA** | [github.com/Junjue-Wang/LoveDA](https://github.com/Junjue-Wang/LoveDA) | building, road, water, vegetation |
| **Roboflow** | [universe.roboflow.com](https://universe.roboflow.com) | Pre-formatted, easiest |

### Step 2: Train on Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `colab_train.py`
3. Set runtime to **GPU (T4)**
4. Run the cells (training takes ~2-3 hours)

### Step 3: Download Weights

After training, download `best.pt` from:
```
/content/geoquery_runs/geoquery_yolo/weights/best.pt
```

Place it here:
```
geoquery/finetune/best.pt
```

### Step 4: Enable in GeoQuery

Edit `config.py`:
```python
USE_YOLO = True
```

Restart the app — it'll now use your fine-tuned model! The sidebar
will show "🎯 YOLOv8 (fine-tuned)" to confirm.

## Files

| File | Purpose |
|------|---------|
| `colab_train.py` | Complete training script for Google Colab |
| `best.pt` | Your trained weights (you create this) |

## For Your Technical Report

Document these in your report:
- **Before**: mAP of zero-shot Grounding DINO on your test images
- **After**: mAP of fine-tuned YOLOv8 on the same test images
- **Per-class AP**: Show which classes improved most
- **Training details**: Dataset, epochs, augmentation, hardware

The `colab_train.py` script prints all these metrics after training.
