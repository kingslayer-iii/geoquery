# GeoQuery

**Author:** Priyanshu
**Target Environment:** Apple Silicon (M4, 16GB Unified Memory)

A chat-based web application for natural language understanding of RGB aerial imagery. Built to solve the IIT Ropar Mock Inter IIT Tech Meet 15.0 GeoQuery problem statement.

## 🌟 Key Features & Innovations

This solution goes beyond the basic requirements by implementing a highly optimized **Hybrid Pipeline** and all three Bonus Innovation features:

1. **Hybrid Object Detection (Innovation 🏆):**
   - **Zero-Shot Foundation (Grounding DINO):** Handles open-vocabulary targets like "buildings", "roads", "water body", "vegetation", and "open ground" seamlessly without any training data.
   - **Fine-Tuned YOLOv8 (VisDrone):** Aerial vehicles are notoriously difficult for zero-shot models. To solve this, I fine-tuned a custom YOLOv8-nano model on the VisDrone dataset (merging car/truck/bus into `vehicle`). The system dynamically routes `vehicle` queries to YOLO (executing in milliseconds) and other classes to DINO, ensuring maximum precision and speed.

2. **Grounded Visual Question Answering (VQA):**
   - **Deterministic Intent Router:** Instead of blindly trusting a VLM to count (which they fail at), a custom NLP intent router classifies queries. It extracts numeric counts and binary (yes/no) presence directly from the physical detection bounding boxes. Generative VQA (BLIP) is only utilized for true attribute questions (color/size).

3. **Interactive Object Queries (Bonus 🏆):**
   - Users can physically click on any generated bounding box in the UI. The application maps the $(x, y)$ coordinates to the underlying object and immediately generates a localized description (color/size/type) for that specific instance.

4. **Multi-Turn Context (Bonus 🏆):**
   - The intent router persists conversation history. You can ask "How many vehicles?" followed by "How many of those are red?", and the system correctly resolves the pronoun "those" to the previous vehicle target.

5. **PDF Export (Bonus 🏆):**
   - Users can download a full session report containing the original image, the annotated image, the auto-caption, and the complete chat transcript in a clean PDF format.

6. **Fully Local & Private:**
   - Runs 100% locally using PyTorch MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration. No paid APIs or external server dependencies.

---

## 🛠️ Setup & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/kingslayer-iii/geoquery.git
cd geoquery

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (MPS support is native on Apple Silicon)
pip install -r requirements.txt
```

### 2. Running the Application
```bash
# Launch the Streamlit server
streamlit run app.py
```
*Note: The first run will securely download the model weights (~1.5GB) from Hugging Face into your local cache. All subsequent runs execute completely offline.*

---

## 📂 Project Architecture

```
geoquery/
├── app.py                  Main Streamlit UI and application orchestrator
├── config.py               Configuration file (model IDs, thresholds, toggle YOLO)
├── models/
│   ├── detector.py         Grounding DINO wrapper for zero-shot classes
│   ├── yolo_detector.py    Custom YOLOv8 wrapper + Hybrid Routing logic
│   ├── captioner.py        Salesforce BLIP wrapper for auto-captioning
│   └── vqa.py              BLIP-VQA handling attribute and fallback queries
├── utils/
│   ├── intent_router.py    NLP router with pronoun resolution and chat history context
│   ├── image_utils.py      Image validation, resizing, bounding box drawing tools
│   └── report.py           PDF Generation module
└── finetune/               Scripts used for the YOLOv8 VisDrone fine-tuning pipeline
```

## ⚙️ Performance Notes (M4 Chip)
- The system defaults to `blip-image-captioning-large` and `Grounding DINO tiny`.
- PyTorch MPS is explicitly enabled. Some fallback operations will run silently on the CPU if Apple's Metal backend does not yet support them. 
- Memory consumption sits around 4-5GB, comfortably fitting inside the 16GB unified memory without swapping, resulting in fast, fluid interactions.
