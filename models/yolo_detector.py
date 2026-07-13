"""
YOLOv8-based object detector for GeoQuery.

Drop-in replacement for the Grounding DINO detector when you have
fine-tuned YOLOv8 weights. Provides significantly better accuracy on
the 6 GeoQuery target classes after fine-tuning on aerial imagery.

Usage:
  1. Train YOLOv8 using finetune/colab_train.py
  2. Place best.pt in finetune/best.pt
  3. Set USE_YOLO = True in config.py
"""

from __future__ import annotations

import os
from typing import List

from PIL import Image

from config import (
    YOLO_WEIGHTS,
    YOLO_CONFIDENCE,
    YOLO_IOU_THRESHOLD,
    TARGET_CLASSES,
)
from utils.image_utils import Detection


class YOLODetector:
    def __init__(self, weights_path: str = YOLO_WEIGHTS, device: str | None = None):
        from ultralytics import YOLO
        from utils.image_utils import get_device

        self.device = device or get_device()

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"YOLO weights not found at '{weights_path}'. "
                f"Train a model using finetune/colab_train.py first, "
                f"then place best.pt in the finetune/ directory."
            )

        self.model = YOLO(weights_path)
        self._class_names = self.model.names  # {0: 'building', 1: 'road', ...}

    def detect(
        self,
        image: Image.Image,
        prompt: str | None = None,       # ignored — YOLO doesn't use text prompts
        box_threshold: float = YOLO_CONFIDENCE,
        text_threshold: float = 0.0,     # ignored — kept for API compatibility
    ) -> List[Detection]:
        """Run YOLOv8 inference on the image.

        The `prompt` and `text_threshold` parameters are accepted but ignored,
        so this class is a drop-in replacement for ObjectDetector (Grounding DINO).
        """
        results = self.model.predict(
            source=image,
            conf=box_threshold,
            iou=YOLO_IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
                confidence = float(box.conf[0])

                # Snap to GeoQuery class names (in case training used different casing)
                clean_name = self._snap_to_class(cls_name)
                if clean_name is None:
                    continue

                # Filter by target class if a prompt specifies one
                if prompt:
                    target = prompt.rstrip(".").strip().lower()
                    if target and target != clean_name:
                        continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        label=clean_name,
                        confidence=confidence,
                        box=(x1, y1, x2, y2),
                    )
                )

        return detections

    def detect_multipass(
        self,
        image: Image.Image,
        classes: List[str] | None = None,
        box_threshold: float = YOLO_CONFIDENCE,
        text_threshold: float = 0.0,
    ) -> List[Detection]:
        """For API compatibility with ObjectDetector. YOLO detects all classes
        in a single pass, so this just calls detect() directly."""
        return self.detect(image, box_threshold=box_threshold)

    @staticmethod
    def _snap_to_class(raw_label: str) -> str | None:
        raw = raw_label.lower().strip()
        for cls in TARGET_CLASSES:
            if cls == raw or cls in raw or raw in cls:
                return cls
        return None

class HybridDetector:
    """Routes detections between YOLO (for fine-tuned classes) and Grounding DINO (for zero-shot classes)."""
    def __init__(self, yolo_detector: YOLODetector):
        from models.detector import ObjectDetector
        self.yolo = yolo_detector
        self.dino = ObjectDetector()
        
        # We know YOLO was only trained on "vehicle"
        self.yolo_classes = set(self.yolo._class_names.values())
        if "vehicle" not in self.yolo_classes:
            self.yolo_classes.add("vehicle") # Fallback safety

    def detect(self, image: Image.Image, prompt: str | None = None, box_threshold: float = 0.25, text_threshold: float = 0.25) -> List[Detection]:
        # If no prompt, use YOLO (which will just return vehicles) and DINO for everything else.
        # But usually we have a prompt from VQA.
        target = prompt.rstrip(".").strip().lower() if prompt else ""
        from config import TARGET_CLASSES
        
        clean_target = None
        for cls in TARGET_CLASSES:
            if cls == target or cls in target or target in cls:
                clean_target = cls
                break
                
        # If the target is one of the YOLO classes (e.g. vehicle), use YOLO!
        if clean_target in self.yolo_classes:
            from config import YOLO_CONFIDENCE
            return self.yolo.detect(image, prompt=prompt, box_threshold=YOLO_CONFIDENCE)
            
        # Otherwise fallback to Grounding DINO
        return self.dino.detect(image, prompt=prompt, box_threshold=box_threshold, text_threshold=text_threshold)

    def detect_multipass(self, image: Image.Image, classes: List[str] | None = None, box_threshold: float = 0.25, text_threshold: float = 0.25) -> List[Detection]:
        all_dets = []
        if classes is None:
            classes = TARGET_CLASSES
            
        for cls in classes:
            all_dets.extend(self.detect(image, prompt=f"{cls}.", box_threshold=box_threshold, text_threshold=text_threshold))
            
        # NMS to deduplicate
        return self._nms(all_dets)

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        if len(detections) <= 1:
            return detections
        
        from utils.spatial import box_iou
        from config import NMS_IOU_THRESHOLD
        
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in sorted_dets:
            if all(box_iou(det.box, k.box) < NMS_IOU_THRESHOLD for k in kept):
                kept.append(det)
        return kept
