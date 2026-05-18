"""Barcode + QR decoding from a tag crop.

zxing-cpp (Apache-2.0): EAN-13/UPC/Code128 + QR, fast, permissive.
OpenCV WeChatQRCode (Apache-2.0): CNN detector + super-resolution, much
stronger on small/blurry QR — used as a QR fallback IF its model files are
present (downloaded by scripts/download_weights.sh; absent -> skipped).

Robot frames are often motion-blurred, so we try several enhancements and,
upstream, several frames (decode_retry_topk) — one good frame is enough
because barcode is the golden dedup key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

try:
    import zxingcpp
    _HAS_ZXING = True
except Exception:  # pragma: no cover
    _HAS_ZXING = False

_BARCODE_FORMATS = {"EAN13", "EAN8", "UPCA", "UPCE", "Code128", "Code39",
                    "ITF", "DataBar", "DataBarExpanded"}


@dataclass
class CodeResult:
    barcode: Optional[str] = None       # 1D product code (EAN-13 etc.)
    qr_payload: Optional[str] = None    # raw QR text
    raw: Optional[list] = None


def _variants(crop: np.ndarray, upscale: float):
    yield crop
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yield g
    if upscale and upscale > 1.0:
        yield cv2.resize(g, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_CUBIC)
    yield cv2.threshold(g, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # mild unsharp — helps slightly defocused codes
    yield cv2.addWeighted(g, 1.5, cv2.GaussianBlur(g, (0, 0), 3), -0.5, 0)


class CodeReader:
    def __init__(self, cfg: dict, models_dir: str = "models"):
        d = cfg["decode"]
        self.use_zxing = bool(d["use_zxing"]) and _HAS_ZXING
        self.upscale = float(d["upscale_for_decode"])
        self.wechat = None
        if d.get("use_wechat_qr"):
            self.wechat = self._load_wechat(models_dir)

    @staticmethod
    def _load_wechat(models_dir: str):
        p = os.path.join(models_dir, "wechat_qr")
        files = [os.path.join(p, f) for f in (
            "detect.prototxt", "detect.caffemodel",
            "sr.prototxt", "sr.caffemodel")]
        if all(os.path.exists(f) for f in files):
            try:
                return cv2.wechat_qrcode.WeChatQRCode(*files)
            except Exception:
                return None
        return None

    def read(self, crop_bgr: np.ndarray) -> CodeResult:
        res = CodeResult(raw=[])
        if crop_bgr is None or crop_bgr.size == 0:
            return res
        if self.use_zxing:
            for img in _variants(crop_bgr, self.upscale):
                try:
                    for r in zxingcpp.read_barcodes(img):
                        fmt = str(r.format).split(".")[-1]
                        txt = r.text
                        res.raw.append((fmt, txt))
                        if fmt == "QRCode" and not res.qr_payload:
                            res.qr_payload = txt
                        elif fmt in _BARCODE_FORMATS and not res.barcode:
                            res.barcode = txt
                except Exception:
                    continue
                if res.barcode and res.qr_payload:
                    return res
        if self.wechat is not None and not res.qr_payload:
            try:
                texts, _ = self.wechat.detectAndDecode(crop_bgr)
                if texts:
                    res.qr_payload = texts[0]
                    res.raw.append(("QRCode:wechat", texts[0]))
            except Exception:
                pass
        return res

    def read_many(self, crops: List[np.ndarray]) -> CodeResult:
        """Try crops in order (best first); merge first barcode + first QR."""
        merged = CodeResult(raw=[])
        for c in crops:
            r = self.read(c)
            merged.raw.extend(r.raw or [])
            if r.barcode and not merged.barcode:
                merged.barcode = r.barcode
            if r.qr_payload and not merged.qr_payload:
                merged.qr_payload = r.qr_payload
            if merged.barcode and merged.qr_payload:
                break
        return merged
