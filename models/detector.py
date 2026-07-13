"""
Zero-shot object detection via Grounding DINO.

Why zero-shot instead of a fine-tuned YOLO: our six classes are natural-language
phrases, and Grounding DINO was trained specifically to localize arbitrary text
phrases in an image with no per-class training data. This lets us support exactly
the class set the problem statement defines without collecting/annotating a custom
dataset — while leaving fine-tuning as an available upgrade for the "innovation"
scoring component (see README, "Optional: fine-tuning").

Key improvements over the baseline:
  - NMS deduplication to remove near-duplicate boxes
  - Per-class threshold support for classes that are harder to detect
  - Multi-pass detection (one pass per class) for improved accuracy
"""

from __future__ import annotations

from typing import List

from PIL import Image

from config import (
    DETECTOR_MODEL_ID,
    DETECTION_PROMPT,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    NMS_IOU_THRESHOLD,
    PER_CLASS_THRESHOLDS,
    TARGET_CLASSES,
)
from utils.image_utils import Detection, get_device
from utils.spatial import box_iou


class ObjectDetector:
    def __init__(self, model_id: str = DETECTOR_MODEL_ID, device: str | None = None):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.device = device or get_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self._torch = torch

    def detect(
        self,
        image: Image.Image,
        prompt: str = DETECTION_PROMPT,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> List[Detection]:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        # Use text_labels if available (transformers v4.51+), fall back to labels.
        labels_key = "text_labels" if "text_labels" in results else "labels"

        detections: List[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"], results[labels_key]):
            clean_label = self._snap_to_class(label)
            if clean_label is None:
                continue
            # Apply per-class threshold if defined.
            threshold = PER_CLASS_THRESHOLDS.get(clean_label, box_threshold)
            if float(score) < threshold:
                continue
            detections.append(
                Detection(
                    label=clean_label,
                    confidence=float(score),
                    box=tuple(box.tolist()),
                )
            )

        # NMS to remove near-duplicate boxes.
        detections = self._nms(detections)
        return detections

    def detect_multipass(
        self,
        image: Image.Image,
        classes: List[str] | None = None,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> List[Detection]:
        """Run separate detection passes for each class and merge results.

        Grounding DINO can get confused when given a long multi-phrase prompt
        (e.g., all 6 classes at once). Running one class at a time improves
        recall and precision, at the cost of ~6x inference time. Recommended
        for the initial full-image scan on upload; single-class queries
        already use a single-class prompt.
        """
        target_classes = classes or TARGET_CLASSES
        all_detections: List[Detection] = []

        for cls in target_classes:
            prompt = f"{cls}."
            threshold = PER_CLASS_THRESHOLDS.get(cls, box_threshold)
            dets = self.detect(image, prompt=prompt, box_threshold=threshold, text_threshold=text_threshold)
            all_detections.extend(dets)

        # Global NMS across all classes (in case different class passes
        # detect the same region with different labels).
        all_detections = self._nms(all_detections)
        return all_detections

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Greedy NMS: sort by confidence, suppress lower-confidence boxes
        that overlap above NMS_IOU_THRESHOLD with any kept box."""
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in sorted_dets:
            if all(box_iou(det.box, k.box) < NMS_IOU_THRESHOLD for k in kept):
                kept.append(det)
        return kept

    @staticmethod
    def _snap_to_class(raw_label: str) -> str | None:
        raw = raw_label.lower().strip()
        for cls in TARGET_CLASSES:
            if cls in raw or raw in cls:
                return cls
        return None
