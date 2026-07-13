"""
Grounded Visual Question Answering.

Design: rather than sending every question straight to a generative VQA
model, we use the intent router (utils/intent_router.py) to decide HOW to
answer:

  numeric   -> count matching Detection objects directly (exact, auditable)
  binary    -> check for presence/absence of matching detections; falls back
               to BLIP-VQA for binary questions not tied to one of the 6
               classes (e.g. "is it daytime?")
  attribute -> if a target class + detections exist, crop the highest-
               confidence box and read color/size off the crop; handles
               "largest"/"smallest" superlatives; falls back to BLIP-VQA
               for attributes it can't compute directly
  coverage  -> compute what fraction of the image is covered by a class
  spatial   -> filter detections by spatial region and answer
  describe  -> generate a focused caption for a specific region or object
  general   -> BLIP-VQA on the full image
  detect    -> handled by the caller (app.py) by re-running detection with
               a narrowed prompt; not answered here

This hybrid keeps counting and presence-checks reliable — generative VLMs
are known to be weak at exact counting — while still handling truly
open-ended questions via the VQA model.
"""

from __future__ import annotations

import re
from typing import List

from PIL import Image

from config import VQA_MODEL_ID, TARGET_CLASSES
from utils.image_utils import Detection, crop_box, dominant_color_name, relative_size_label, get_device
from utils.intent_router import classify, Intent
from utils.spatial import (
    filter_by_region,
    coverage_fraction,
    largest_detection,
    smallest_detection,
    box_area,
    match_region,
)


class VisualQA:
    def __init__(self, model_id: str = VQA_MODEL_ID, device: str | None = None):
        import torch
        from transformers import BlipProcessor, BlipForQuestionAnswering

        self.device = device or get_device()
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self._torch = torch

    # ---- public entry point -------------------------------------------------
    def answer(self, image: Image.Image, query: str, detections: List[Detection]) -> str:
        intent = classify(query)

        if not intent.in_scope and intent.target_class is None:
            return (
                "That object isn't one of the six classes this system supports "
                f"({', '.join(TARGET_CLASSES)}). Try asking about one of those instead."
            )

        if intent.query_type == "numeric":
            return self._answer_numeric(intent, detections, image)
        if intent.query_type == "binary":
            return self._answer_binary(intent, detections, image, query)
        if intent.query_type == "attribute":
            return self._answer_attribute(intent, detections, image, query)
        if intent.query_type == "coverage":
            return self._answer_coverage(intent, detections, image)
        if intent.query_type == "spatial":
            return self._answer_spatial(intent, detections, image, query)
        if intent.query_type == "describe":
            return self._answer_describe(intent, detections, image, query)

        # "general" (and "detect" questions that fall through here) -> raw VQA
        return self._blip_vqa(image, query)

    # ---- grounded handlers ----------------------------------------------
    def _answer_numeric(self, intent: Intent, detections: List[Detection], image: Image.Image) -> str:
        if intent.target_class is None:
            return (
                "I can count objects for the six supported classes "
                f"({', '.join(TARGET_CLASSES)}) — could you specify which one?"
            )

        # Filter by region if spatial reference is present.
        dets = detections
        region_note = ""
        if intent.region:
            dets = filter_by_region(dets, intent.region, image.width, image.height)
            region_note = f" in the {intent.region}"

        matches = [d for d in dets if d.label == intent.target_class]
        n = len(matches)
        noun = intent.target_class + ("s" if n != 1 else "")

        if n == 0:
            return f"I don't detect any {noun}{region_note} in this image."

        # Build a confidence summary.
        confs = [d.confidence for d in matches]
        avg_conf = sum(confs) / len(confs)
        return (
            f"I detected **{n} {noun}**{region_note} in this image "
            f"(avg confidence: {avg_conf:.0%})."
        )

    def _answer_binary(self, intent: Intent, detections: List[Detection], image: Image.Image, query: str) -> str:
        if intent.target_class is not None:
            # Filter by region if applicable.
            dets = detections
            region_note = ""
            if intent.region:
                dets = filter_by_region(dets, intent.region, image.width, image.height)
                region_note = f" in the {intent.region}"

            present = any(d.label == intent.target_class for d in dets)
            if present:
                count = sum(1 for d in dets if d.label == intent.target_class)
                return (
                    f"**Yes**, there {'is' if count == 1 else 'are'} "
                    f"{count} {intent.target_class}{'s' if count > 1 else ''} "
                    f"visible{region_note} in this image."
                )
            else:
                return f"**No**, I don't detect any {intent.target_class}{region_note} in this image."

        # Not tied to one of our 6 classes (e.g. "is it daytime?") -> generative fallback
        return self._blip_vqa(image, query)

    def _answer_attribute(self, intent: Intent, detections: List[Detection], image: Image.Image, query: str) -> str:
        if intent.target_class is not None:
            matches = [d for d in detections if d.label == intent.target_class]
            if not matches:
                return f"I don't see any {intent.target_class} in this image to describe."

            q_lower = query.lower()

            # Superlative queries: largest / smallest
            if any(w in q_lower for w in ["largest", "biggest"]):
                det = largest_detection(detections, intent.target_class)
                if det is None:
                    return f"I don't detect any {intent.target_class} to compare."
                crop = crop_box(image, det.box)
                color = dominant_color_name(crop)
                size = relative_size_label(det.box, image)
                return (
                    f"The **largest {intent.target_class}** (confidence {det.confidence:.0%}) "
                    f"is {size} relative to the full image, and appears mostly **{color}**."
                )

            if any(w in q_lower for w in ["smallest", "tiniest"]):
                det = smallest_detection(detections, intent.target_class)
                if det is None:
                    return f"I don't detect any {intent.target_class} to compare."
                crop = crop_box(image, det.box)
                color = dominant_color_name(crop)
                size = relative_size_label(det.box, image)
                return (
                    f"The **smallest {intent.target_class}** (confidence {det.confidence:.0%}) "
                    f"is {size} relative to the full image, and appears mostly **{color}**."
                )

            best = max(matches, key=lambda d: d.confidence)
            crop = crop_box(image, best.box)

            if "colour" in q_lower or "color" in q_lower:
                color = dominant_color_name(crop)
                return (
                    f"The {intent.target_class} (confidence {best.confidence:.0%}) "
                    f"appears mostly **{color}**."
                )
            if "big" in q_lower or "size" in q_lower:
                size = relative_size_label(best.box, image)
                return (
                    f"That {intent.target_class} looks **{size}** relative to the full image "
                    f"(bbox covers {box_area(best.box) / (image.width * image.height):.1%} of the image)."
                )

            # Other attribute phrasing -> let the VQA model describe the crop
            return self._blip_vqa(crop, query)

        return self._blip_vqa(image, query)

    def _answer_coverage(self, intent: Intent, detections: List[Detection], image: Image.Image) -> str:
        if intent.target_class is None:
            # Coverage across all detected objects.
            frac = coverage_fraction(detections, None, image.width, image.height)
            return (
                f"Detected objects cover approximately **{frac:.0%}** of the image area "
                f"(across {len(detections)} detections from all classes)."
            )

        frac = coverage_fraction(detections, intent.target_class, image.width, image.height)
        count = sum(1 for d in detections if d.label == intent.target_class)

        if count == 0:
            return f"I don't detect any {intent.target_class} in this image, so coverage is 0%."

        return (
            f"**{intent.target_class.capitalize()}** covers approximately **{frac:.0%}** of the image area "
            f"(based on {count} detected region{'s' if count > 1 else ''})."
        )

    def _answer_spatial(self, intent: Intent, detections: List[Detection], image: Image.Image, query: str) -> str:
        region = intent.region
        if not region:
            return self._blip_vqa(image, query)

        region_dets = filter_by_region(detections, region, image.width, image.height)

        if intent.target_class:
            class_dets = [d for d in region_dets if d.label == intent.target_class]
            n = len(class_dets)
            if n > 0:
                return (
                    f"**Yes**, there {'is' if n == 1 else 'are'} **{n} {intent.target_class}"
                    f"{'s' if n > 1 else ''}** in the {region} of the image."
                )
            else:
                return f"**No**, I don't detect any {intent.target_class} in the {region} of the image."
        else:
            if not region_dets:
                return f"I don't detect any of the six supported classes in the {region} of the image."
            summary_parts = []
            from collections import Counter
            counts = Counter(d.label for d in region_dets)
            for cls, cnt in counts.most_common():
                summary_parts.append(f"{cnt} {cls}{'s' if cnt > 1 else ''}")
            return (
                f"In the **{region}** of the image, I detect: {', '.join(summary_parts)}."
            )

    def _answer_describe(self, intent: Intent, detections: List[Detection], image: Image.Image, query: str) -> str:
        """Generate a focused description — either of a specific class region
        or a spatial region of the image."""
        if intent.target_class:
            matches = [d for d in detections if d.label == intent.target_class]
            if not matches:
                return f"I don't see any {intent.target_class} in this image to describe."
            best = max(matches, key=lambda d: d.confidence)
            crop = crop_box(image, best.box)
            desc = self._blip_vqa(crop, f"Describe this {intent.target_class} in detail.")
            color = dominant_color_name(crop)
            size = relative_size_label(best.box, image)
            return (
                f"The {intent.target_class} (confidence {best.confidence:.0%}): "
                f"{desc}. It appears mostly **{color}** and is **{size}** relative to the full image."
            )

        if intent.region:
            region_dets = filter_by_region(detections, intent.region, image.width, image.height)
            if not region_dets:
                # Fall back to VQA on the full image.
                return self._blip_vqa(image, query)
            from collections import Counter
            counts = Counter(d.label for d in region_dets)
            parts = [f"{cnt} {cls}{'s' if cnt > 1 else ''}" for cls, cnt in counts.most_common()]
            return (
                f"In the **{intent.region}** of the image, I see: {', '.join(parts)}."
            )

        # Generic describe -> full-image VQA
        return self._blip_vqa(image, query)

    # ---- generative fallback ---------------------------------------------
    def _blip_vqa(self, image: Image.Image, query: str) -> str:
        inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(out[0], skip_special_tokens=True).strip().capitalize()
