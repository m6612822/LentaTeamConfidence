"""Automatic tag colour -> english label (GT `color` is english lowercase:
red, yellow, ...). Decided by the MEDIAN hue/sat of background pixels
(text/white/glare/dark masked out) -> robust to printed text and reflections.
Vocabulary + HSV ranges come from config/patterns.yaml (data-calibrated).
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _in_hue(h: float, ranges: List) -> bool:
    # ranges may be [lo,hi] or [[lo,hi],[lo,hi]]
    if ranges and isinstance(ranges[0], (list, tuple)):
        return any(lo <= h <= hi for lo, hi in ranges)
    return ranges[0] <= h <= ranges[1]


def detect_color(crop_bgr: np.ndarray, cfg: dict) -> str:
    """Return an english colour label, or "" if the crop is unusable
    (colour is virtually always present on a tag -> rarely "нет")."""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    labels = cfg["patterns"]["color_labels"]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # keep coloured background: drop near-white, dark, and washed-out pixels
    keep = (S > 45) & (V > 50) & (V < 245)
    if keep.sum() < max(20, 0.02 * keep.size):
        # low-saturation tag -> likely white
        for lab in labels:
            if lab["name"] == "white":
                return "white"
        return ""
    h_med = float(np.median(H[keep]))
    s_med = float(np.median(S[keep]))
    v_med = float(np.median(V[keep]))
    best, best_pri = "", -1.0
    for lab in labels:
        if "h" in lab and not _in_hue(h_med, lab["h"]):
            continue
        if "s_min" in lab and s_med < lab["s_min"]:
            continue
        if "s_max" in lab and s_med > lab["s_max"]:
            continue
        if "v_min" in lab and v_med < lab["v_min"]:
            continue
        # prefer the more saturated/specific match
        pri = s_med
        if pri > best_pri:
            best, best_pri = lab["name"], pri
    return best
