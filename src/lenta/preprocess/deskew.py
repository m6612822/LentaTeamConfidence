"""Crop orientation for OCR. Tags are ~90° rotated in the native frame
(camera mounted 90° CCW). Default: rotate the CROP 90° CW. If OCR is weak,
try all 4 right-angle rotations and keep the most confident — robust to the
barrel distortion that tilts tags unpredictably.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

_ROT = {0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate(crop: np.ndarray, deg: int) -> np.ndarray:
    op = _ROT.get(deg)
    return crop if op is None else cv2.rotate(crop, op)


def best_for_ocr(crop: np.ndarray, ocr_engine, cfg: dict):
    """Return (oriented_crop, ocr_lines). One OCR call in the common case;
    up to 4 only when the primary orientation reads poorly."""
    if not cfg["preprocess"].get("rotate_crop_for_ocr", True):
        lines = ocr_engine.read(crop)
        return crop, lines
    primary = rotate(crop, 90)
    lines = ocr_engine.read(primary)
    if ocr_engine.mean_conf(lines) >= 0.55 and len(lines) >= 2:
        return primary, lines
    best_c, best_l, best_s = primary, lines, _score(ocr_engine, lines)
    for deg in (0, 180, 270):
        c = rotate(crop, deg)
        l = ocr_engine.read(c)
        s = _score(ocr_engine, l)
        if s > best_s:
            best_c, best_l, best_s = c, l, s
    return best_c, best_l


def _score(eng, lines: List) -> float:
    if not lines:
        return 0.0
    return eng.mean_conf(lines) * sum(len(l.text) for l in lines)
