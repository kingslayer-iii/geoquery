"""
Image captioning via BLIP (Salesforce/blip-image-captioning-large).

BLIP was chosen over a heavier VLM (LLaVA) for the default caption path
because it's fast enough to run automatically on every upload with low
latency on a single consumer GPU. If your GPU has headroom and you want
richer, more detailed captions, see README -> "Upgrading to LLaVA".
"""

from __future__ import annotations

from PIL import Image

from config import CAPTIONER_MODEL_ID
from utils.image_utils import get_device


class ImageCaptioner:
    def __init__(self, model_id: str = CAPTIONER_MODEL_ID, device: str | None = None):
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.device = device or get_device()
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self._torch = torch

    def caption(self, image: Image.Image, max_new_tokens: int = 40) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text.strip().capitalize()
