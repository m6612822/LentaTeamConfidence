"""Best-frame selection: pick the frame where a tag is recognised most
clearly (moment of FINAL determination, not first appearance).

quality = w_lap * f(LaplacianVar)            # focus blur
        + w_fft * FFT_high_freq_ratio        # motion blur (directional)
        + w_det * detector_score
with a multiplicative penalty if the bbox touches the frame border
(truncated tag). Laplacian alone underestimates motion blur from the moving
robot, so the FFT term is required, not optional.
"""
from __future__ import annotations

import heapq
from typing import List, Tuple

import cv2
import numpy as np

from lenta.types import BBox, Observation, TagTrack

_LAP_SAT = 300.0  # saturating constant for Laplacian variance -> [0,1)


def laplacian_score(gray: np.ndarray) -> float:
    v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return v / (v + _LAP_SAT)


def fft_hf_ratio(gray: np.ndarray, low_frac: float = 0.125) -> float:
    """Energy fraction outside a central low-frequency square. Sharp/clean
    images keep more high-frequency energy; motion blur collapses it."""
    g = gray
    if max(g.shape) > 256:
        s = 256.0 / max(g.shape)
        g = cv2.resize(g, (max(1, int(g.shape[1] * s)),
                           max(1, int(g.shape[0] * s))))
    f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    ry, rx = max(1, int(h * low_frac)), max(1, int(w * low_frac))
    total = float(mag.sum()) + 1e-9
    low = float(mag[cy - ry:cy + ry, cx - rx:cx + rx].sum())
    return max(0.0, min(1.0, 1.0 - low / total))


def crop_quality(crop_bgr: np.ndarray, det_score: float, truncated: bool,
                 w_lap: float, w_fft: float, w_det: float,
                 border_penalty: float, w_area: float = 0.0,
                 area_frac: float = 0.0) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return -1.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # sqrt(area_frac) so a close pass (large crop) dominates — codes/fine
    # text are only decodable when the tag is big enough.
    q = (w_lap * laplacian_score(gray)
         + w_fft * fft_hf_ratio(gray)
         + w_det * float(det_score)
         + w_area * float(area_frac) ** 0.5)
    if truncated:
        q *= border_penalty
    return q


def update_track(track: TagTrack, frame_bgr: np.ndarray, bbox: BBox,
                 det_score: float, frame_idx: int, ts_ms: float,
                 cfg: dict) -> None:
    """Add one observation. Keeps only the top-K crops (decode retry budget)
    so memory stays O(K) per track regardless of track length."""
    bf = cfg["best_frame"]
    h, w = frame_bgr.shape[:2]
    b = bbox.clip(w, h)
    x0, y0, x1, y1 = int(b.x0), int(b.y0), int(b.x1), int(b.y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    crop = frame_bgr[y0:y1, x0:x1]
    q = crop_quality(crop, det_score, bbox.touches_border(w, h),
                     bf["w_lap"], bf["w_fft"], bf["w_det"],
                     bf["border_penalty"],
                     bf.get("w_area", 0.0), bbox.area / float(w * h))
    track.observations.append(Observation(frame_idx, ts_ms, bbox, q))
    k = int(bf["decode_retry_topk"])
    # min-heap of (quality, tiebreak, crop); keep K best
    tie = len(track.observations)
    if len(track.candidate_crops) < k:
        heapq.heappush(track.candidate_crops, (q, tie, crop.copy()))
    elif q > track.candidate_crops[0][0]:
        heapq.heapreplace(track.candidate_crops, (q, tie, crop.copy()))
    if q > track.best_quality:
        track.best_quality = q
        track.best_bbox = bbox
        track.best_ts_ms = ts_ms


def best_crops(track: TagTrack) -> List[Tuple[float, np.ndarray]]:
    """Top-K crops, best first (for OCR on #1, decode retry on the rest)."""
    return [(q, c) for q, _, c in
            sorted(track.candidate_crops, key=lambda t: -t[0])]
