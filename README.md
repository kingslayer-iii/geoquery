# GeoQuery

A chat-based app for natural language understanding of RGB imagery — captioning,
zero-shot object detection (6 classes), and grounded visual question answering.
Built for the IIT Ropar Mock Inter IIT Tech Meet 15.0 GeoQuery problem statement.

## 1. Setup (Apple Silicon — M4)

```bash
cd geoquery
python3 -m venv venv
source venv/bin/activate

# On Apple Silicon, the standard PyPI wheel already includes MPS (Metal) GPU
# support — no separate CUDA-style install step needed.
pip install -r requirements.txt
```

Your M4's GPU will be picked up automatically as the `mps` device (see
`utils/image_utils.get_device()`), no config changes needed.

**Memory note**: 16GB is unified between the OS, CPU, and GPU. The default
model set (Grounding DINO tiny + BLIP-large captioning + BLIP-VQA base) is
roughly 4GB of weights, which fits comfortably, but close memory-heavy apps
(Chrome with many tabs, etc.) before your first run. If you see MPS
out-of-memory errors or things feel sluggish, switch to the smaller
captioning model noted in `config.py` (`blip-image-captioning-base` instead
of `-large`).

**MPS-specific gotcha**: a small number of PyTorch ops aren't implemented on
Apple's Metal backend yet. `app.py` already sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` at startup, which makes those specific ops
silently run on CPU instead of crashing — you don't need to do anything, but
it's why you'll occasionally see it dip in and out of full GPU speed.

**Fanless-chassis note**: the Air has no active cooling. A single upload
(caption + detect) or chat query is quick, but if you're stress-testing with
many images back-to-back for your report's evaluation section, expect some
thermal throttling on longer runs — not a bug, just Air hardware.


## 2. Run

```bash
streamlit run app.py
```

First run will download ~1.5GB of model weights (Grounding DINO tiny + BLIP
captioning + BLIP-VQA) from Hugging Face — this needs internet access once;
after that they're cached locally (`~/.cache/huggingface`) and load offline.

## 3. Project structure

```
geoquery/
├── app.py                  Streamlit UI + chat orchestration
├── config.py                All model names, classes, thresholds — edit here
├── models/
│   ├── detector.py           Grounding DINO wrapper (zero-shot detection)
│   ├── captioner.py          BLIP captioning wrapper
│   └── vqa.py                Grounded VQA: routes to deterministic answers
│                              or falls back to BLIP-VQA
└── utils/
    ├── image_utils.py        Validation, resizing, box drawing, color extraction
    ├── intent_router.py      Classifies each query: numeric/binary/attribute/
    │                          detect/general, and which of the 6 classes it targets
    └── report.py              PDF export of annotated image + transcript
```

## 4. How each PS requirement is satisfied

| PS Section | Requirement | Where it's implemented |
|---|---|---|
| 2.1 | Auto-caption on upload | `app.py` upload handler -> `ImageCaptioner.caption()` |
| 2.2 | Detection w/ boxes, labels, confidence | `ObjectDetector.detect()` + `draw_detections()` |
| 2.3 | Binary / numeric / attribute VQA | `VisualQA.answer()` routes via `intent_router.classify()` |
| 3.1 | .jpg/.png only, max 1024px, graceful errors | `validate_extension()`, `load_and_prepare()` |
| 3.2 | Open-source, locally hosted, no paid APIs | Grounding DINO + BLIP, all run locally |
| 5 | Unified chat, session history, error handling | `app.py` single `st.chat_input`, `st.session_state.chat_history` |
| 5 (bonus) | Multi-turn context | Detections persist in `st.session_state` across turns |
| 5 (bonus) | Exportable PDF report | `utils/report.py` + download button |

Not yet implemented (see "Extending" below): object-level click-to-query,
SAM-based segmentation instead of boxes.

## 5. Design rationale (for your technical report)

**Why Grounding DINO instead of a fine-tuned YOLO?**
The six classes are natural-language phrases, several of which (e.g. "water
body", "open ground") don't map cleanly onto any standard pretrained
detector's label set. Grounding DINO takes an image + free-text prompt and
localizes arbitrary phrases zero-shot, so we get exactly these six classes
with no annotation or training required. The trade-off: it's slower per
image than YOLO and won't outperform a properly fine-tuned detector on
accuracy — worth mentioning as a limitation in your report.

**Why not just ask a VLM every question directly?**
Generative VLMs are known to be unreliable at exact counting and precise
localization, even though they're good at open-ended description. GeoQuery
uses `intent_router.py` to classify each query, then answers numeric and
presence/absence questions **directly from the detection list** (an exact
`len()` count, not a model guess) and only calls the generative VQA model
for genuinely open-ended questions. This "grounding" pattern — using a
structured, verifiable data source to answer precise questions and a
generative model only for language understanding — is exactly the kind of
system design the PS background section is asking for, and is worth a
dedicated paragraph in your report.

**Known limitation to disclose:** the intent router is regex/keyword-based,
not a trained classifier. It's fast and fully auditable (a judge can see
exactly why a query was routed a certain way), but will misclassify unusual
phrasings. A stretch goal is swapping it for a small local LLM prompt-based
classifier — mention this as future work.

## 6. Optional: fine-tuning (innovation scoring)

The PS explicitly rewards fine-tuning in the innovation component. If you
have time:
- Fine-tune Grounding DINO or a YOLOv8 model on a remote-sensing dataset
  with overlapping classes — DOTA, iSAID, or LoveDA are good starting points
  (all public, all have building/road/vehicle/vegetation-type classes).
- Even a few hundred hand-labeled boxes on images similar to your demo set,
  fine-tuning just the detection head, can measurably improve precision.
- Document before/after mAP in your report — quantified improvement is what
  scores here, not just "we fine-tuned it."

## 7. Extending further

- **Object-level click queries**: Streamlit doesn't support native image
  click coordinates well; consider `streamlit-image-coordinates` package,
  or switch the image panel to an HTML canvas via `st.components.v1.html`
  if you want this bonus feature.
- **Segmentation instead of boxes**: run SAM on each Grounding DINO box as a
  point/box prompt to get pixel masks — improves the "largest building"
  style attribute questions since you get a precise silhouette, not just a
  rectangle.
- **Upgrading captions/VQA to LLaVA**: swap `CAPTIONER_MODEL_ID` /
  `VQA_MODEL_ID` logic for a single LLaVA call (e.g. a quantized
  `llava-hf/llava-1.5-7b-hf`) — stronger free-form reasoning, at the cost of
  latency. On your 16GB unified-memory M4, this only really works with a
  4-bit/8-bit quantized checkpoint; a full-precision 7B model won't fit
  alongside the OS and other models. Test this as a separate experiment
  rather than swapping it in right before a demo.

## 8. Troubleshooting

- **MPS out of memory / everything feels slow**: switch `CAPTIONER_MODEL_ID`
  in `config.py` to `"Salesforce/blip-image-captioning-base"` (smaller model,
  see the memory note in Section 1), close other memory-heavy apps, and/or
  reduce `MAX_RESOLUTION`.
- **A crash mentioning an op "not implemented for MPS"**: this shouldn't
  happen since `app.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` at startup —
  if you do hit one, it means that env var wasn't picked up (check nothing
  imports `torch` before `app.py`'s first few lines run) or upgrade torch,
  since MPS op coverage improves every release.
- **Detections look wrong / too many false positives**: raise
  `BOX_THRESHOLD` in `config.py` (try 0.4–0.5).
- **Detections missing obvious objects**: lower `BOX_THRESHOLD` /
  `TEXT_THRESHOLD` slightly, or check the class snapping logic in
  `detector.py::_snap_to_class` — Grounding DINO sometimes returns partial
  phrase matches.
- **First run is slow**: that's the one-time model download (~4GB); after
  that they load from the local Hugging Face cache (`~/.cache/huggingface`),
  works offline from then on.
