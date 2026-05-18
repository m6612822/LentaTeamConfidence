"""End-to-end orchestrator.

Modes (detector is pluggable):
  gt       — oracle boxes from the sibling reference CSV. Isolates
             OCR/QR/field/нет-empty quality (the real risk) for fast
             self-eval iteration. One GT row per tag -> already deduped.
  onnx     — trained lightweight detector (production / Unlabeled videos).
  zeroshot — Grounding DINO fallback (no training).

Flow (onnx/zeroshot): stream frames -> detect -> IoU track -> on track
close pick best frame -> OCR + code decode on best crop(s) -> field
assembly -> reconcile (нет/empty) -> barcode-first dedup -> CSV rows.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np

from lenta.assemble.color_detect import detect_color
from lenta.assemble.fields import FieldExtractor
from lenta.assemble.reconcile import build_row
from lenta.aggregate.dedup import deduplicate
from lenta.decode.code_reader import CodeReader
from lenta.io.video_reader import VideoReader
from lenta.preprocess.fusion import fuse_tag
from lenta.ocr.engine import OCREngine
from lenta.preprocess.deskew import best_for_ocr
from lenta.schema import CSV_COLUMNS
from lenta.types import BBox, TagTrack


def _fnum(x):
    try:
        return float(str(x).replace(",", "."))
    except (ValueError, TypeError):
        return None


def _crop(img: np.ndarray, b: BBox, pad: int = 6):
    h, w = img.shape[:2]
    b = b.clip(w, h)
    x0, y0 = max(0, int(b.x0) - pad), max(0, int(b.y0) - pad)
    x1, y1 = min(w, int(b.x1) + pad), min(h, int(b.y1) + pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return img[y0:y1, x0:x1]


class Pipeline:
    def __init__(self, cfg: dict, models_dir: str = "models"):
        self.cfg = cfg
        self.ocr = OCREngine(cfg, models_dir)
        self.codes = CodeReader(cfg, models_dir)
        self.fields = FieldExtractor(cfg)

    # ---- per-tag field assembly (shared by all modes) ----
    def _assemble(self, crop_best: np.ndarray, crops_all: List[np.ndarray],
                  meta: Dict[str, object]) -> Dict[str, str]:
        oriented, lines = best_for_ocr(crop_best, self.ocr, self.cfg)
        code = self.codes.read_many(crops_all or [crop_best])
        visual = self.fields.extract(lines)
        color = detect_color(crop_best, self.cfg)
        healthy = self.ocr.mean_conf(lines) >= self.cfg["ocr"].get(
            "healthy_crop_confidence", 0.5) or bool(code.barcode
                                                    or code.qr_payload)
        return build_row(visual, code.qr_payload, code.barcode,
                         color, meta, healthy, self.cfg)

    # ---- GT-oracle mode (eval / field tuning) ----
    def run_gt(self, video: str, ref_csv: str,
               out_filename: Optional[str] = None) -> List[Dict[str, str]]:
        vr = VideoReader(video)
        with open(ref_csv, encoding="utf-8") as fh:
            gt = list(csv.DictReader(fh))
        by_ts: Dict[int, list] = {}
        for r in gt:
            ts = _fnum(r.get("frame_timestamp"))
            if ts is not None:
                by_ts.setdefault(round(ts), []).append(r)
        fn = out_filename or os.path.basename(video)
        use_fusion = self.cfg.get("fusion", {}).get("enabled", False)
        rows: List[Dict[str, str]] = []
        for ts, grp in sorted(by_ts.items()):
            fr = vr.frame_at_ms(float(ts))
            if fr is None:
                continue
            for r in grp:
                x0, y0 = _fnum(r["x_min"]), _fnum(r["y_min"])
                x1, y1 = _fnum(r["x_max"]), _fnum(r["y_max"])
                if None in (x0, y0, x1, y1):
                    continue
                b = BBox(x0, y0, x1, y1)
                single = _crop(fr.image, b)
                if single is None:
                    continue
                best = single
                if use_fusion:
                    fused = fuse_tag(video, float(ts), b, self.cfg)
                    if fused is not None and fused.size:
                        best = fused
                meta = {"filename": fn, "frame_timestamp": ts,
                        "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}
                rows.append(self._assemble(best, [best, single], meta))
        return rows

    # ---- detector mode (production) ----
    def run_detect(self, video: str, detector,
                   out_filename: Optional[str] = None) -> List[Dict[str, str]]:
        from lenta.aggregate.tag_track import best_crops
        from lenta.track.linker import IoUTracker

        vr = VideoReader(video)
        fn = out_filename or os.path.basename(video)
        tracker = IoUTracker(self.cfg)
        stride = int(self.cfg["video"]["base_stride"])
        closed: List[TagTrack] = []
        for fr in vr.frames():
            if fr.idx % stride != 0:
                continue
            dets = detector.detect(fr.image, fr.idx, fr.ts_ms)
            closed += tracker.update(dets, fr.image, fr.ts_ms)
        closed += tracker.flush()

        for t in closed:
            crops = [c for _, c in best_crops(t)]
            if not crops:
                continue
            code = self.codes.read_many(crops)
            t.barcode = code.barcode
        kept = deduplicate(closed, self.cfg)

        rows: List[Dict[str, str]] = []
        for t in kept:
            crops = [c for _, c in best_crops(t)]
            if not crops or t.best_bbox is None:
                continue
            b = t.best_bbox
            meta = {"filename": fn, "frame_timestamp": round(t.best_ts_ms),
                    "x_min": b.x0, "y_min": b.y0,
                    "x_max": b.x1, "y_max": b.y1}
            rows.append(self._assemble(crops[0], crops, meta))
        return rows
