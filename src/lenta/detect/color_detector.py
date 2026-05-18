"""Classical colour+geometry price-tag detector — no GPU, no training,
instant on CPU (also ideal for the live demo / edge).

Lenta shelf tags are highly saturated red/orange/yellow rectangles on the
shelf rail with white text. We threshold those hues in HSV, close gaps,
find rectangular blobs in a plausible size/aspect range, and NMS them.
This exploits the strong visual prior of THIS setting instead of a heavy
learned detector. Thresholds live in config (data-calibrated).
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from lenta.types import BBox, Detection


def _nms(boxes: List[BBox], iou_thr: float) -> List[BBox]:
    if not boxes:
        return []
    b = sorted(boxes, key=lambda x: x.area, reverse=True)
    keep: List[BBox] = []
    for cand in b:
        if all(cand.iou(k) < iou_thr for k in keep):
            keep.append(cand)
    return keep


class ColorTagDetector:
    def __init__(self, cfg: dict):
        d = cfg.get("color_detector", {})
        self.min_area_frac = float(d.get("min_area_frac", 0.0008))
        self.max_area_frac = float(d.get("max_area_frac", 0.15))
        self.min_ar = float(d.get("min_aspect", 0.20))
        self.max_ar = float(d.get("max_aspect", 5.0))
        self.min_white = float(d.get("min_white_frac", 0.06))
        self.max_white = float(d.get("max_white_frac", 0.85))
        self.nms_iou = float(d.get("nms_iou", 0.3))
        self.pad = float(d.get("pad_frac", 0.04))

    def detect(self, image_bgr: np.ndarray, frame_idx: int,
               ts_ms: float) -> List[Detection]:
        H, W = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # red (wraps 0/180) + orange + yellow, all high-saturation
        m1 = cv2.inRange(hsv, (0, 90, 70), (15, 255, 255))
        m2 = cv2.inRange(hsv, (160, 90, 70), (180, 255, 255))
        m3 = cv2.inRange(hsv, (15, 110, 90), (40, 255, 255))   # orange/yellow
        mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(
                                    cv2.MORPH_RECT, (3, 3)))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(W * H)
        boxes: List[BBox] = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            af = (w * h) / frame_area
            if not (self.min_area_frac <= af <= self.max_area_frac):
                continue
            ar = w / float(h) if h else 0.0
            if not (self.min_ar <= ar <= self.max_ar):
                continue
            # solidity: tag is a filled rectangle, not a thin ring
            if cv2.contourArea(c) < 0.35 * w * h:
                continue
            # white-text discrimination: a price tag has white text/digits
            # inside; a solid red product box does not.
            roi = hsv[y:y + h, x:x + w]
            white = ((roi[..., 2] > 170) & (roi[..., 1] < 70)).mean()
            if not (self.min_white <= white <= self.max_white):
                continue
            px, py = w * self.pad, h * self.pad
            boxes.append(BBox(x - px, y - py, x + w + px, y + h + py)
                         .clip(W, H))
        return [Detection(b, 1.0, frame_idx, ts_ms)
                for b in _nms(boxes, self.nms_iou)]
