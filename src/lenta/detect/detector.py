"""Lightweight detector inference (ONNX Runtime, CPU).

Single class: price_tag. The trained student (D-FINE-S, Apache-2.0; or
MIT-YOLOv9) is exported to ONNX by scripts/autolabel_distill.py /
the training step and placed at detector.weights.

Output decoding supports the two dominant ONNX conventions so the same
runtime works whichever exporter is used (verified once the trained model
is exported):
  A) single tensor (1,N,6) = [x0,y0,x1,y1,score,cls]  (NMS baked in)
  B) three tensors boxes(N,4)/scores(N,)/labels(N,)   (DETR-style export)
A Grounding-DINO zero-shot path is the documented fallback if no weights
exist (implemented in detect/zeroshot.py, loaded lazily).
"""
from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np

from lenta.types import BBox, Detection


def letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    dy, dx = (size - nh) // 2, (size - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


class ONNXDetector:
    def __init__(self, cfg: dict):
        import onnxruntime as ort
        d = cfg["detector"]
        self.size = int(d["input_size"])
        self.score_thr = float(d["score_threshold"])
        self.weights = d["weights"]
        if not os.path.exists(self.weights):
            raise FileNotFoundError(
                f"detector weights not found: {self.weights} "
                f"(run scripts/download_weights.sh or train first; "
                f"or set detector.zeroshot_fallback: true)")
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg.get("run", {}).get("threads", 4))
        self.sess = ort.InferenceSession(
            self.weights, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def _decode(self, outs, r, dx, dy, W, H) -> List[Tuple[BBox, float]]:
        res: List[Tuple[BBox, float]] = []

        def unpad(x0, y0, x1, y1):
            return (BBox((x0 - dx) / r, (y0 - dy) / r,
                         (x1 - dx) / r, (y1 - dy) / r).clip(W, H))

        if len(outs) == 1:                       # convention A
            arr = outs[0]
            arr = arr[0] if arr.ndim == 3 else arr
            for row in arr:
                if row.shape[0] < 6:
                    continue
                x0, y0, x1, y1, sc = row[:5]
                if float(sc) >= self.score_thr:
                    res.append((unpad(x0, y0, x1, y1), float(sc)))
        else:                                    # convention B
            boxes = outs[0].reshape(-1, 4)
            scores = outs[1].reshape(-1)
            for (x0, y0, x1, y1), sc in zip(boxes, scores):
                if float(sc) >= self.score_thr:
                    res.append((unpad(x0, y0, x1, y1), float(sc)))
        return res

    def detect(self, image_bgr: np.ndarray, frame_idx: int,
               ts_ms: float) -> List[Detection]:
        H, W = image_bgr.shape[:2]
        canvas, r, dx, dy = letterbox(image_bgr, self.size)
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        outs = self.sess.run(None, {self.inp: blob})
        return [Detection(b, s, frame_idx, ts_ms)
                for b, s in self._decode(outs, r, dx, dy, W, H)]
