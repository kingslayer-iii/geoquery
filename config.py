"""
Central configuration for GeoQuery.

Edit model names, thresholds, and the class list here — nothing else in the
codebase should hardcode these values. This makes swapping models (e.g.
grounding-dino-tiny -> grounding-dino-base for higher accuracy) a one-line change.
"""

# ---- Target classes (Section 4 of the problem statement) -------------------
TARGET_CLASSES = [
    "building",
    "road",
    "vehicle",
    "vegetation",
    "water body",
    "open ground",
]

# Synonyms so free-form user queries ("cars", "trucks", "river", "trees")
# get correctly mapped onto one of the six official classes.
CLASS_SYNONYMS = {
    "building": [
        "building", "buildings", "structure", "structures", "house", "houses",
        "rooftop", "rooftops", "roof", "roofs", "apartment", "apartments",
        "skyscraper", "tower", "towers", "warehouse", "shed",
    ],
    "road": [
        "road", "roads", "street", "streets", "highway", "highways",
        "pavement", "lane", "lanes", "path", "paths", "intersection",
        "crossroad", "avenue", "boulevard",
    ],
    "vehicle": [
        "vehicle", "vehicles", "car", "cars", "truck", "trucks", "bus",
        "buses", "van", "vans", "motorcycle", "motorbike", "auto",
        "automobile", "transport", "parked car", "parked cars",
    ],
    "vegetation": [
        "vegetation", "tree", "trees", "grass", "forest", "crop", "crops",
        "field", "fields", "plants", "plant", "greenery", "green area",
        "garden", "gardens", "shrub", "shrubs", "bush", "bushes", "foliage",
        "canopy", "lawn",
    ],
    "water body": [
        "water body", "water bodies", "water", "river", "rivers", "lake",
        "lakes", "pond", "ponds", "sea", "stream", "streams", "canal",
        "reservoir", "creek", "waterway",
    ],
    "open ground": [
        "open ground", "bare land", "barren land", "sand", "sandy",
        "unpaved", "empty land", "vacant land", "bare soil", "dirt",
        "clearing", "open area", "open space", "wasteland", "ground",
        "bare patch", "empty lot",
    ],
}

# Grounding DINO expects a lowercase, period-separated prompt string.
DETECTION_PROMPT = ". ".join(TARGET_CLASSES) + "."

# ---- Model choices -----------------------------------------------------
# All open-source, locally hosted — satisfies Section 3.2 of the PS
# (no paid commercial vision APIs allowed as the inference backbone).

# Zero-shot / open-vocabulary object detector.
# Using grounding-dino-base for better accuracy (~340M params).
# Switch to "IDEA-Research/grounding-dino-tiny" (~170M) if memory is tight.
DETECTOR_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Image captioning model.
CAPTIONER_MODEL_ID = "Salesforce/blip-image-captioning-large"

# General VQA model, used as a fallback for open-ended questions that
# aren't about counting or a specific detected object's attribute.
VQA_MODEL_ID = "Salesforce/blip-vqa-base"

# ---- Detection thresholds ------------------------------------------------
BOX_THRESHOLD = 0.35    # min confidence to keep a detected box
TEXT_THRESHOLD = 0.25   # min token-matching confidence for the text label

# Per-class threshold overrides: some classes (like "open ground" or
# "water body") are harder to detect and benefit from a lower threshold.
PER_CLASS_THRESHOLDS = {
    "building":    0.35,
    "road":        0.30,
    "vehicle":     0.35,
    "vegetation":  0.30,
    "water body":  0.28,
    "open ground": 0.25,
}

# NMS (Non-Maximum Suppression) IoU threshold — detections with IoU above
# this with a higher-confidence detection are suppressed. Prevents near-
# duplicate boxes that Grounding DINO often produces.
NMS_IOU_THRESHOLD = 0.5

# ---- Input constraints (Section 3.1 of the PS) ---------------------------
MAX_RESOLUTION = 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ---- Device ---------------------------------------------------------------
# Resolved automatically at runtime in utils/image_utils.get_device()
# (CUDA > MPS/Apple Silicon > CPU). PREFERRED_DEVICE below is unused by
# default and kept only for reference/manual override.
PREFERRED_DEVICE = "mps"

# ---- Fine-tuned YOLO detector (Section 6 — innovation scoring) -----------
# Set USE_YOLO = True after training your own YOLOv8 model using
# finetune/colab_train.py. This replaces Grounding DINO with your
# fine-tuned model for significantly better accuracy on the 6 classes.
USE_YOLO = True
YOLO_WEIGHTS = "finetune/best.pt"     # path to your trained weights
YOLO_CONFIDENCE = 0.40                 # min confidence for YOLO detections
YOLO_IOU_THRESHOLD = 0.45             # NMS IoU threshold for YOLO

# ---- Memory-constrained setups (e.g. 16GB unified memory on Apple Silicon) -
# If you hit MPS out-of-memory errors or things feel too slow, switch the
# captioner to the smaller base model (drops from ~470M to ~223M params):
#   CAPTIONER_MODEL_ID = "Salesforce/blip-image-captioning-base"
# And/or switch detector back to tiny:
#   DETECTOR_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

