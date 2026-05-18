"""Neural super-resolution (Real-ESRGAN general-x4-v3, ONNX, BSD-3).

~4.9 MB, dynamic input, fp32, ×4 — runs on the installed onnxruntime
(CPU), light enough for edge. Trained on REAL degradation (compression +
blur), i.e. exactly this footage's failure mode — unlike bicubic-trained
ESPCN. Used as the final stage of fusion: median-fuse aligned frames at
native resolution (kills compression noise) → neural ×4 → OCR.

Honest note: proven that SR cannot resurrect destroyed QR/barcode patterns
(WeChat-SR 0/275). Expected gain is on printed text/large digits, not codes.
Tiled to bound CPU memory on large fused crops.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


class SuperRes:
    def __init__(self, cfg: dict, models_dir: str = "models"):
        s = cfg.get("superres", {})
        self.enabled = bool(s.get("enabled", False))
        self.scale = int(s.get("scale", 4))
        self.tile = int(s.get("tile", 256))
        self.overlap = int(s.get("overlap", 16))
        self.max_side = int(s.get("max_input_side", 1400))
        self._sess = None
        self._in = None
        path = s.get("model_path") or os.path.join(
            models_dir, "superres", "realesr-general-x4v3.onnx")
        if self.enabled and os.path.exists(path):
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = int(
                    cfg.get("run", {}).get("threads", 4))
                self._sess = ort.InferenceSession(
                    path, so, providers=["CPUExecutionProvider"])
                self._in = self._sess.get_inputs()[0].name
            except Exception:
                self._sess = None
        if self.enabled and self._sess is None:
            # model absent/failed -> disable gracefully (pipeline still runs)
            self.enabled = False

    def _infer(self, rgb01: np.ndarray) -> np.ndarray:
        x = np.transpose(rgb01, (2, 0, 1))[None].astype(np.float32)
        y = self._sess.run(None, {self._in: x})[0][0]
        return np.clip(np.transpose(y, (1, 2, 0)), 0.0, 1.0)

    def upscale(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return ×scale BGR, or None if disabled/unavailable."""
        if not self.enabled or self._sess is None or bgr is None \
                or bgr.size == 0:
            return None
        h, w = bgr.shape[:2]
        # cap input so a huge fused crop doesn't blow CPU memory
        if max(h, w) > self.max_side:
            r = self.max_side / float(max(h, w))
            bgr = cv2.resize(bgr, (int(w * r), int(h * r)),
                             interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        s, t, ov = self.scale, self.tile, self.overlap
        out = np.zeros((h * s, w * s, 3), np.float32)
        for y0 in range(0, h, t - ov):
            for x0 in range(0, w, t - ov):
                y1, x1 = min(h, y0 + t), min(w, x0 + t)
                sr = self._infer(rgb[y0:y1, x0:x1])
                # blend overlap region by simple max-coverage write
                out[y0 * s:y1 * s, x0 * s:x1 * s] = sr
        return cv2.cvtColor((out * 255.0).astype(np.uint8),
                            cv2.COLOR_RGB2BGR)
