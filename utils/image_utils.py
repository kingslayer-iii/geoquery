"""
Image handling utilities: validation, resizing, bounding-box rendering,
and dominant-color extraction for attribute questions.

Kept free of any ML model imports so it can be unit-tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import MAX_RESOLUTION, ALLOWED_EXTENSIONS


@dataclass
class Detection:
    """A single detected object, in a model-agnostic format."""
    label: str
    confidence: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords


# A richer set of named colors for attribute questions.
_NAMED_COLORS = {
    "red": (200, 40, 40), "dark red": (120, 20, 20),
    "orange": (230, 140, 40), "brown": (110, 70, 40),
    "yellow": (220, 200, 60), "golden": (200, 170, 50),
    "tan": (200, 180, 140), "cream": (240, 225, 200),
    "green": (60, 140, 60), "dark green": (30, 90, 40),
    "lime": (130, 200, 50), "olive": (110, 120, 50),
    "teal": (50, 140, 140), "cyan": (70, 200, 200),
    "blue": (50, 90, 180), "light blue": (140, 180, 220),
    "dark blue": (20, 40, 110), "navy": (20, 30, 80),
    "purple": (120, 50, 160), "pink": (220, 130, 160),
    "magenta": (180, 50, 130),
    "gray": (130, 130, 130), "dark gray": (70, 70, 70),
    "light gray": (190, 190, 190), "silver": (200, 200, 210),
    "white": (230, 230, 225), "black": (30, 30, 30),
    "beige": (215, 200, 175), "khaki": (190, 180, 130),
    "maroon": (90, 20, 30), "rust": (170, 80, 40),
}


def get_device() -> str:
    """Resolve the best available torch device: CUDA > MPS (Apple Silicon) > CPU.
    Imports torch lazily so this module stays importable in environments
    without torch installed."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def validate_extension(filename: str) -> Optional[str]:
    """Returns an error message string if the extension is not allowed,
    otherwise None."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"⚠️ Unsupported file type '.{ext}'. Please upload a **.jpg** or **.png** "
            f"image (only standard RGB images are supported)."
        )
    return None


def load_and_prepare(file_bytes: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Loads image bytes, converts to RGB, and resizes down to MAX_RESOLUTION
    if needed. Returns (image, error_message); exactly one will be None."""
    import io
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return None, "⚠️ Could not read this file as an image. Please upload a valid .jpg or .png."

    w, h = img.size
    if max(w, h) > MAX_RESOLUTION:
        scale = MAX_RESOLUTION / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img, None


def draw_detections(image: Image.Image, detections: List[Detection]) -> Image.Image:
    """Returns a copy of `image` with bounding boxes, labels, and confidence
    scores overlaid. Uses class-specific colors and improved label styling."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Distinct, high-contrast palette for the 6 classes.
    CLASS_PALETTE = {
        "building":    (230, 70, 70),    # red
        "road":        (70, 130, 230),   # blue
        "vehicle":     (255, 200, 50),   # yellow
        "vegetation":  (50, 200, 80),    # green
        "water body":  (50, 180, 220),   # cyan
        "open ground": (200, 130, 60),   # brown/orange
    }
    fallback_palette = [
        (230, 60, 60), (60, 140, 230), (60, 200, 120),
        (230, 180, 40), (170, 90, 220), (240, 130, 60),
    ]

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()

    for i, det in enumerate(detections):
        color = CLASS_PALETTE.get(det.label, fallback_palette[i % len(fallback_palette)])
        x1, y1, x2, y2 = det.box

        # Draw box with rounded-feel thick outline.
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Semi-transparent label background.
        tag = f"{det.label} {det.confidence:.0%}"
        text_bbox = draw.textbbox((0, 0), tag, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - th - 8)
        draw.rectangle([x1, label_y, x1 + tw + 10, label_y + th + 6], fill=color)
        draw.text((x1 + 5, label_y + 2), tag, fill=(255, 255, 255), font=font)

    return annotated


def crop_box(image: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    return image.crop((x1, y1, x2, y2))


def dominant_color_name(crop: Image.Image) -> str:
    """Estimates the dominant color of an image crop and maps it to the
    nearest named color. Used to answer 'what colour is X' attribute
    questions with a deterministic, cheap heuristic rather than relying
    solely on the VLM (which is often unreliable on color for small crops)."""
    if crop.width == 0 or crop.height == 0:
        return "unknown"

    small = crop.resize((32, 32))
    arr = np.array(small).reshape(-1, 3).astype(float)

    # k-means with k=1 is just the mean, but a quick 3-cluster pass and
    # picking the largest cluster is more robust to shadow/edge pixels.
    from sklearn.cluster import KMeans
    k = min(3, len(arr))
    if k < 1:
        return "unknown"
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(arr)
    counts = np.bincount(km.labels_)
    dominant = km.cluster_centers_[np.argmax(counts)]

    best_name, best_dist = "unknown", float("inf")
    for name, rgb in _NAMED_COLORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(dominant, rgb))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name


def relative_size_label(box: Tuple[float, float, float, float], image: Image.Image) -> str:
    """Cheap heuristic for 'how big is X' style attribute questions:
    expresses a box's area as a fraction of the image and buckets it."""
    x1, y1, x2, y2 = box
    box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    frac = box_area / (image.width * image.height)
    if frac < 0.01:
        return "very small"
    elif frac < 0.05:
        return "small"
    elif frac < 0.15:
        return "medium-sized"
    elif frac < 0.30:
        return "large"
    else:
        return "very large"
