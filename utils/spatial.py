"""
Spatial reasoning utilities for bounding-box geometry.

Used by the VQA module to answer location-aware questions like
"Is there a road in the lower half?" or "What fraction of the
image is vegetation?", and by the intent router to detect spatial
query patterns.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from utils.image_utils import Detection


# ---------------------------------------------------------------------------
# Region definitions — divide the image into named spatial zones.
# All regions are expressed as (x_frac_start, y_frac_start, x_frac_end,
# y_frac_end), where 0.0 is top-left and 1.0 is bottom-right.
# ---------------------------------------------------------------------------
REGIONS = {
    "upper half":   (0.0, 0.0, 1.0, 0.5),
    "lower half":   (0.0, 0.5, 1.0, 1.0),
    "left half":    (0.0, 0.0, 0.5, 1.0),
    "right half":   (0.5, 0.0, 1.0, 1.0),
    "upper left":   (0.0, 0.0, 0.5, 0.5),
    "upper right":  (0.5, 0.0, 1.0, 0.5),
    "lower left":   (0.0, 0.5, 0.5, 1.0),
    "lower right":  (0.5, 0.5, 1.0, 1.0),
    "center":       (0.25, 0.25, 0.75, 0.75),
    "top":          (0.0, 0.0, 1.0, 0.33),
    "bottom":       (0.0, 0.67, 1.0, 1.0),
    "left":         (0.0, 0.0, 0.33, 1.0),
    "right":        (0.67, 0.0, 1.0, 1.0),
}

# Synonyms for matching user queries to region names.
REGION_SYNONYMS = {
    "upper half": ["upper half", "top half", "upper portion", "top portion"],
    "lower half": ["lower half", "bottom half", "lower portion", "bottom portion"],
    "left half": ["left half", "left side", "left portion"],
    "right half": ["right half", "right side", "right portion"],
    "upper left": ["upper left", "top left", "top-left"],
    "upper right": ["upper right", "top right", "top-right"],
    "lower left": ["lower left", "bottom left", "bottom-left"],
    "lower right": ["lower right", "bottom right", "bottom-right"],
    "center": ["center", "centre", "middle"],
    "top": ["top", "upper part"],
    "bottom": ["bottom", "lower part"],
    "left": ["left", "left part"],
    "right": ["right", "right part"],
}


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Returns the (cx, cy) center of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_area(box: Tuple[float, float, float, float]) -> float:
    """Returns the pixel area of a bounding box."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Intersection-over-Union of two boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_in_region(
    box: Tuple[float, float, float, float],
    region: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    overlap_threshold: float = 0.5,
) -> bool:
    """Returns True if at least `overlap_threshold` fraction of the box's
    area falls within the given region (specified as fractional coords)."""
    rx1 = region[0] * img_w
    ry1 = region[1] * img_h
    rx2 = region[2] * img_w
    ry2 = region[3] * img_h

    ix1 = max(box[0], rx1)
    iy1 = max(box[1], ry1)
    ix2 = min(box[2], rx2)
    iy2 = min(box[3], ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    ba = box_area(box)
    return (inter / ba) >= overlap_threshold if ba > 0 else False


def filter_by_region(
    detections: List[Detection],
    region_name: str,
    img_w: int,
    img_h: int,
) -> List[Detection]:
    """Returns detections whose center (or majority area) falls within the
    named region."""
    region = REGIONS.get(region_name)
    if region is None:
        return detections
    return [d for d in detections if box_in_region(d.box, region, img_w, img_h)]


def match_region(query: str) -> Optional[str]:
    """Scans the query for any spatial region reference and returns the
    canonical region name, or None."""
    q = query.lower()
    for region_name, synonyms in REGION_SYNONYMS.items():
        for syn in synonyms:
            if syn in q:
                return region_name
    return None


def coverage_fraction(
    detections: List[Detection],
    target_class: Optional[str],
    img_w: int,
    img_h: int,
) -> float:
    """Estimates what fraction of the image is covered by detections of the
    given class. Uses simple bbox union (no overlap dedup for speed)."""
    total_area = img_w * img_h
    if total_area == 0:
        return 0.0

    matched = [d for d in detections if target_class is None or d.label == target_class]
    if not matched:
        return 0.0

    # Approximate: sum of box areas clipped to image, capped at 1.0.
    covered = sum(box_area(d.box) for d in matched)
    return min(1.0, covered / total_area)


def sort_by_size(
    detections: List[Detection],
    descending: bool = True,
) -> List[Detection]:
    """Returns detections sorted by bounding-box area."""
    return sorted(detections, key=lambda d: box_area(d.box), reverse=descending)


def largest_detection(
    detections: List[Detection],
    target_class: Optional[str] = None,
) -> Optional[Detection]:
    """Returns the largest detection (by box area), optionally filtered to a class."""
    candidates = [d for d in detections if target_class is None or d.label == target_class]
    if not candidates:
        return None
    return max(candidates, key=lambda d: box_area(d.box))


def smallest_detection(
    detections: List[Detection],
    target_class: Optional[str] = None,
) -> Optional[Detection]:
    """Returns the smallest detection (by box area), optionally filtered to a class."""
    candidates = [d for d in detections if target_class is None or d.label == target_class]
    if not candidates:
        return None
    return min(candidates, key=lambda d: box_area(d.box))
