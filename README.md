# GeoQuery

A Streamlit web app for natural language understanding of RGB aerial imagery. Upload an image, get an automatic caption and object detections, then ask anything about the scene in plain English.

Built for the IIT Ropar Mock Inter IIT Tech Meet 15.0 GeoQuery problem statement.

---

## Features

### Core Pipeline

- **Automatic Captioning** — Salesforce BLIP generates a natural language description of the uploaded scene on every image load.
- **Multi-class Object Detection** — Detects and localises six aerial object classes with coloured bounding boxes and confidence scores.
- **Natural Language Q&A** — Ask counting, presence, attribute, spatial, and coverage questions; the system routes each query to the appropriate handler rather than passing everything blindly to a generative model.

### Supported Classes

| Class | Description |
|-------|-------------|
| `building` | Structures, rooftops, warehouses |
| `road` | Streets, highways, lanes, intersections |
| `vehicle` | Cars, trucks, buses, motorbikes |
| `vegetation` | Trees, grass, crops, forest canopy |
| `water body` | Rivers, lakes, ponds, canals |
| `open ground` | Bare soil, sandy areas, empty lots |

---

## Architecture

```
geoquery/
├── app.py                  Streamlit UI and application orchestrator
├── config.py               Central config — model IDs, thresholds, class list
├── models/
│   ├── detector.py         Grounding DINO wrapper (zero-shot detection)
│   ├── yolo_detector.py    YOLOv8 wrapper + HybridDetector routing logic
│   ├── captioner.py        BLIP image captioning wrapper
│   └── vqa.py              BLIP-VQA for attribute and open-ended queries
├── utils/
│   ├── intent_router.py    Deterministic NLP intent classifier with pronoun resolution
│   ├── image_utils.py      Image I/O, bounding box drawing, color extraction
│   ├── spatial.py          Spatial region parsing and IoU utilities
│   └── report.py           PDF session report generator
└── finetune/               YOLOv8 fine-tuning scripts (VisDrone dataset)
```

### Detection — Hybrid Pipeline

Detection is handled by one of two backends, selected automatically:

**Grounding DINO (default on cloud / new installs)**
Zero-shot, open-vocabulary detector. Runs a separate inference pass per class (`detect_multipass`) to avoid the label-confusion that occurs when all six class names are concatenated into a single prompt. Results from each pass are merged and deduplicated with greedy NMS.

**HybridDetector (local, when `finetune/best.pt` is present)**
Routes `vehicle` queries to a custom YOLOv8-nano model fine-tuned on VisDrone (with `car`, `truck`, and `bus` merged into a single `vehicle` class). All other classes fall through to Grounding DINO. This eliminates the known weakness of zero-shot models on small, densely-packed aerial vehicles while keeping the flexibility of open-vocabulary detection for the remaining classes.

`config.py` auto-detects which backend to use:
```python
USE_YOLO = os.path.exists("finetune/best.pt")  # True locally, False on cloud
```

### Intent Router

Rather than passing every query to the VLM and hoping it counts correctly, `utils/intent_router.py` classifies each query into one of eight types using lightweight regex pattern matching:

| Type | Example | Handler |
|------|---------|---------|
| `detect` | "Mark all water bodies" | Re-runs detection, updates annotated image |
| `numeric` | "How many vehicles?" | `len(detections)` — no hallucination possible |
| `binary` | "Is there a road?" | Boolean check on detection list |
| `coverage` | "What fraction is vegetation?" | Bounding box area ratio |
| `attribute` | "What colour is the largest building?" | K-means dominant color on cropped box |
| `spatial` | "Any vehicles in the lower half?" | Filters detections by region |
| `describe` | "Describe the open ground" | BLIP-VQA on cropped region |
| `general` | Everything else | BLIP-VQA on the full image |

The router also resolves pronouns across turns: asking "How many vehicles?" then "How many of those are red?" correctly maps "those" back to `vehicle` without re-prompting the user.

### Clickable Bounding Boxes

The annotated image is rendered with `streamlit-image-coordinates`. Clicking a bounding box in the UI maps the pixel coordinates back to the underlying `Detection` object and calls the attribute handler on that specific crop — returning colour, size, and a description of exactly what was clicked.

### PDF Report

After a conversation, the session can be exported as a PDF containing the original image, the annotated image, the auto-caption, and the full chat transcript. Generated with ReportLab; no external services required.

---

## Setup

### Requirements

- Python 3.10+
- ~4–5 GB RAM for Grounding DINO base + BLIP-large + BLIP-VQA
- GPU recommended (CUDA or Apple Silicon MPS); CPU inference works but is slow

### Installation

```bash
git clone https://github.com/kingslayer-iii/geoquery.git
cd geoquery

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running

```bash
streamlit run app.py
```

On first run, Hugging Face will download the model weights (~4 GB total) into the local cache (`~/.cache/huggingface`). All subsequent runs are fully offline.

The sidebar shows which device (CUDA / MPS / CPU) and detector backend are active.

---

## Configuration

All tuneable parameters are in [`config.py`](config.py) — nothing is hardcoded elsewhere.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DETECTOR_MODEL_ID` | `IDEA-Research/grounding-dino-base` | Swap to `-tiny` to reduce memory usage |
| `CAPTIONER_MODEL_ID` | `Salesforce/blip-image-captioning-large` | Swap to `-base` to reduce memory usage |
| `BOX_THRESHOLD` | `0.35` | Global minimum detection confidence |
| `PER_CLASS_THRESHOLDS` | see config | Per-class overrides (lower for harder classes) |
| `NMS_IOU_THRESHOLD` | `0.5` | Overlap threshold for suppressing duplicate boxes |
| `MAX_RESOLUTION` | `1024` | Images are downscaled to this before inference |
| `USE_YOLO` | auto | `True` if `finetune/best.pt` exists |

---

## Fine-tuning (Optional)

The `finetune/` directory contains scripts for training a custom YOLOv8-nano model on the VisDrone dataset. This is what produces `finetune/best.pt` — the weights that activate the `HybridDetector`.

```bash
# Train in Google Colab (recommended — free T4 GPU)
# Upload finetune/colab_train.py to a Colab notebook and run it.

# Or train locally (requires a GPU):
cd finetune
python train_local.py
```

After training, place `best.pt` in the `finetune/` directory and restart the app. The hybrid routing activates automatically.

See [`finetune/README.md`](finetune/README.md) for dataset preparation details.

---

## Deployment

The app is deployed on Streamlit Community Cloud. The cloud environment uses CPU-only PyTorch and Grounding DINO (YOLO weights are excluded from the repo via `.gitignore`). The `packages.txt` file installs `libgl1` to satisfy OpenCV's native dependency on Ubuntu.

Live demo: [geoquery.streamlit.app](https://geoquery.streamlit.app)

---

## Performance Notes (Apple Silicon)

On an M4 MacBook with 16 GB unified memory:

- Memory footprint sits around 4–5 GB — comfortably fits without swapping
- MPS acceleration is enabled via `PYTORCH_ENABLE_MPS_FALLBACK=1` (set before any torch import so ops not yet on Metal fall back to CPU silently rather than crashing)
- `detect_multipass` adds ~6x the single-pass inference time but meaningfully improves recall on crowded scenes
- Switching to `grounding-dino-tiny` and `blip-image-captioning-base` halves memory and speeds up inference if needed

---

## License

MIT
