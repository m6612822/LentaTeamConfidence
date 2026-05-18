"""Grounding DINO zero-shot detector (Apache-2.0).

The no-training fallback of the hybrid strategy and the auto-labeler for the
Unlabeled videos. Needs torch+transformers (requirements-autolabel.txt) —
heavy/slow on CPU, so production uses the distilled ONNX detector; this is
for auto-labeling and as a detector-of-last-resort.
"""
from __future__ import annotations

from typing import List

import numpy as np

from lenta.types import BBox, Detection


class GroundingDINODetector:
    def __init__(self, cfg: dict, models_dir: str = "models"):
        try:
            import torch
            from transformers import (AutoModelForZeroShotObjectDetection,
                                      AutoProcessor)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "zeroshot mode needs torch+transformers: "
                "pip install -r requirements-autolabel.txt") from e
        self._torch = torch
        d = cfg["detector"]
        model_id = d.get("zeroshot_model", "IDEA-Research/grounding-dino-tiny")
        self.prompt = d.get("zeroshot_prompt",
                            "price tag . ценник . shelf label .")
        self.box_thr = float(d.get("score_threshold", 0.35))
        self.text_thr = float(d.get("zeroshot_text_threshold", 0.25))
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).eval()

    def detect(self, image_bgr: np.ndarray, frame_idx: int,
               ts_ms: float) -> List[Detection]:
        from PIL import Image

        img = Image.fromarray(image_bgr[:, :, ::-1])
        inp = self.proc(images=img, text=self.prompt, return_tensors="pt")
        with self._torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, inp["input_ids"], box_threshold=self.box_thr,
            text_threshold=self.text_thr,
            target_sizes=[img.size[::-1]])[0]
        dets: List[Detection] = []
        for box, score in zip(res["boxes"].tolist(),
                              res["scores"].tolist()):
            x0, y0, x1, y1 = box
            dets.append(Detection(BBox(x0, y0, x1, y1), float(score),
                                  frame_idx, ts_ms))
        return dets
