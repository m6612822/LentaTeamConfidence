"""OCR via RapidOCR (PP-OCR models on onnxruntime, Apache-2.0).

Chosen over paddlepaddle/paddleocr: same PP-OCR model quality but runs on
the already-installed onnxruntime — far lighter on this 4-core/3GB box and
no paddle/paddlex dependency churn.

Cyrillic: RapidOCR's built-in rec model is ch/en. A Cyrillic PP-OCR rec
model + dict are downloaded by scripts/download_weights.sh and wired via
config (ocr.rec_model_path / ocr.rec_keys_path); if absent we fall back to
the built-in model (still reads digits/latin — i.e. prices, barcode, dates,
discount %, the metric-heavy numeric fields — while product_name needs the
Cyrillic model).
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from lenta.assemble.fields import OcrLine


class OCREngine:
    def __init__(self, cfg: dict, models_dir: str = "models"):
        from rapidocr_onnxruntime import RapidOCR

        o = cfg.get("ocr", {})
        kw = {}
        rec = o.get("rec_model_path")
        keys = o.get("rec_keys_path")
        if rec and os.path.exists(rec):
            kw["rec_model_path"] = rec
            if keys and os.path.exists(keys):
                kw["rec_keys_path"] = keys
        # constrained CPU: limit intra-op threads
        try:
            self.engine = RapidOCR(**kw)
        except Exception:
            self.engine = RapidOCR()
        self.min_conf = float(o.get("min_text_confidence", 0.6))

    @staticmethod
    def _enhance(crop: np.ndarray, target_long: int = 900) -> np.ndarray:
        """Upscale small/blurry tag crops + denoise + unsharp so the OCR
        detector sees enough resolution (4K tag crops are tiny & compressed)."""
        h, w = crop.shape[:2]
        long_side = max(h, w)
        if long_side < target_long:
            s = target_long / float(long_side)
            crop = cv2.resize(crop, (int(w * s), int(h * s)),
                              interpolation=cv2.INTER_CUBIC)
        den = cv2.fastNlMeansDenoisingColored(crop, None, 3, 3, 7, 21)
        blur = cv2.GaussianBlur(den, (0, 0), 3)
        return cv2.addWeighted(den, 1.5, blur, -0.5, 0)

    def read(self, crop_bgr: np.ndarray) -> List[OcrLine]:
        if crop_bgr is None or crop_bgr.size == 0:
            return []
        try:
            result, _ = self.engine(self._enhance(crop_bgr))
        except Exception:
            try:
                result, _ = self.engine(crop_bgr)
            except Exception:
                return []
        lines: List[OcrLine] = []
        for item in result or []:
            box, text, conf = item[0], item[1], float(item[2])
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            lines.append(OcrLine(text=text, conf=conf,
                                 x=float(x), y=float(y),
                                 w=float(w), h=float(h)))
        return lines

    def mean_conf(self, lines: List[OcrLine]) -> float:
        return sum(l.conf for l in lines) / len(lines) if lines else 0.0
