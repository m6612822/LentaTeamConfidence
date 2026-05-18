"""Multi-frame fusion / burst super-resolution for one price tag.

The robot footage is heavily H.264-compressed + motion-blurred, so a single
crop is mostly unreadable (proven: single='149' -> fused readable brand +
price). Each tag is seen across dozens of frames as the robot passes;
tracking it, sub-pixel aligning and robust-fusing averages out compression
noise and motion blur and recovers detail.

Per tag (seed = (ts, bbox), e.g. a GT box or a track's best obs):
  1. follow the tag through a ±window via NCC tracking on DOWNSCALED gray
     (cheap; robot motion is smooth -> small search region)
  2. keep the K sharpest observations
  3. estimate ECC warp at LOW res, apply it to the UPSCALED crop
     (ECC at low res = ~10x cheaper, same alignment)
  4. robust temporal median fuse + mild unsharp -> sharp hi-res crop
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from lenta.io.video_reader import VideoReader
from lenta.types import BBox

_ALIGN_LONG = 360  # px: ECC estimation resolution (speed/accuracy sweet spot)


def _sharp(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _track(vr: VideoReader, ts_ms: float, seed: BBox, win_ms: float,
           max_obs: int, step: int) -> List[np.ndarray]:
    seed_fr = vr.frame_at_ms(ts_ms)
    if seed_fr is None:
        return []
    H, W = seed_fr.image.shape[:2]
    b = seed.clip(W, H)
    x0, y0, x1, y1 = int(b.x0), int(b.y0), int(b.x1), int(b.y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    tmpl = seed_fr.image[y0:y1, x0:x1].copy()
    th, tw = tmpl.shape[:2]
    sc = min(1.0, 240.0 / max(th, tw))               # track at low res
    tmpl_g = cv2.resize(cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY), None,
                        fx=sc, fy=sc)
    obs: List[Tuple[float, np.ndarray]] = [
        (_sharp(cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)), tmpl)]
    last = (x0, y0)
    for i, fr in enumerate(vr.frames_window(ts_ms - win_ms, ts_ms + win_ms)):
        if i % step or abs(fr.ts_ms - seed_fr.ts_ms) < 1.0:
            continue
        mh, mw = int(th * 0.9), int(tw * 0.9)
        sx0, sy0 = max(0, last[0] - mw), max(0, last[1] - mh)
        sx1 = min(W, last[0] + tw + mw)
        sy1 = min(H, last[1] + th + mh)
        region = fr.image[sy0:sy1, sx0:sx1]
        if region.shape[0] < th or region.shape[1] < tw:
            continue
        rg = cv2.resize(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), None,
                        fx=sc, fy=sc)
        if rg.shape[0] < tmpl_g.shape[0] or rg.shape[1] < tmpl_g.shape[1]:
            continue
        res = cv2.matchTemplate(rg, tmpl_g, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx < 0.45:
            continue
        px, py = sx0 + int(ml[0] / sc), sy0 + int(ml[1] / sc)
        last = (px, py)
        crop = fr.image[py:py + th, px:px + tw]
        if crop.shape[:2] != (th, tw):
            continue
        obs.append((_sharp(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), crop))
    obs.sort(key=lambda o: -o[0])
    return [c for _, c in obs[:max_obs]]


def _align_fuse(crops: List[np.ndarray], scale: float) -> Optional[np.ndarray]:
    if not crops:
        return None
    ref = crops[0]
    rh, rw = ref.shape[:2]
    W, H = int(rw * scale), int(rh * scale)
    asc = min(1.0, _ALIGN_LONG / float(max(rh, rw)))
    aw, ah = max(8, int(rw * asc)), max(8, int(rh * asc))
    ref_a = cv2.cvtColor(cv2.resize(ref, (aw, ah)),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
    stack = [cv2.resize(ref, (W, H),
                        interpolation=cv2.INTER_CUBIC).astype(np.float32)]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 1e-3)
    for c in crops[1:]:
        ca = cv2.cvtColor(cv2.resize(c, (aw, ah)),
                          cv2.COLOR_BGR2GRAY).astype(np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(ref_a, ca, warp,
                                            cv2.MOTION_AFFINE, crit, None, 5)
        except cv2.error:
            continue
        if cc < 0.30:
            continue
        # rescale the low-res warp to the upscaled target
        s = scale / asc
        warp_up = warp.copy()
        warp_up[0, 2] *= s
        warp_up[1, 2] *= s
        up = cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)
        up = cv2.warpAffine(up, warp_up, (W, H),
                            flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        stack.append(up.astype(np.float32))
    fused = np.median(np.stack(stack, 0), 0).astype(np.uint8)
    blur = cv2.GaussianBlur(fused, (0, 0), 3)
    return cv2.addWeighted(fused, 1.6, blur, -0.6, 0)


_SR = None  # cached SuperRes (avoid reloading the ONNX per tag)


def _get_sr(cfg: dict):
    global _SR
    if _SR is None:
        from lenta.preprocess.superres import SuperRes
        _SR = SuperRes(cfg)
    return _SR


def fuse_tag(video: str, ts_ms: float, bbox: BBox,
             cfg: dict) -> Optional[np.ndarray]:
    f = cfg.get("fusion", {})
    vr = VideoReader(video)
    crops = _track(vr, ts_ms, bbox,
                   float(f.get("window_ms", 1000)),
                   int(f.get("max_obs", 9)),
                   int(f.get("frame_step", 2)))
    if not crops:
        return None
    sr = _get_sr(cfg) if cfg.get("superres", {}).get("enabled") else None
    if sr is not None and sr.enabled:
        # fuse at NATIVE res (median denoise) -> neural ×4
        fused = _align_fuse(crops, 1.0)
        up = sr.upscale(fused) if fused is not None else None
        if up is not None:
            blur = cv2.GaussianBlur(up, (0, 0), 3)
            return cv2.addWeighted(up, 1.3, blur, -0.3, 0)
        return fused
    return _align_fuse(crops, float(f.get("upscale", 2.5)))
