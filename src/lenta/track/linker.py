"""Lightweight IoU tracker — short-term frame linking only.

Deliberately simple: the robot path is monotonic and final dedup is
barcode-first (see aggregate/dedup.py), so a heavy ReID tracker
(DeepSORT/BoT-SORT) is unnecessary and would only add deps. Greedy IoU
association is enough to group consecutive observations of one tag so we
can pick its best frame.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from lenta.aggregate.tag_track import update_track
from lenta.types import BBox, Detection, TagTrack


class _Active:
    __slots__ = ("track", "last_bbox", "misses", "hits")

    def __init__(self, track: TagTrack, bbox: BBox):
        self.track = track
        self.last_bbox = bbox
        self.misses = 0
        self.hits = 1


class IoUTracker:
    def __init__(self, cfg: dict):
        t = cfg["tracker"]
        self.iou_thr = float(t["iou_match_threshold"])
        self.max_age = int(t["max_age"])
        self.close_after = int(t["close_after_misses"])
        self.min_hits = int(t["min_hits"])
        self.cfg = cfg
        self._active: Dict[int, _Active] = {}
        self._next_id = 0

    def _new(self, det: Detection, frame, ts_ms) -> _Active:
        self._next_id += 1
        tr = TagTrack(track_id=self._next_id)
        a = _Active(tr, det.bbox)
        update_track(tr, frame, det.bbox, det.score,
                     det.frame_idx, ts_ms, self.cfg)
        self._active[self._next_id] = a
        return a

    def update(self, detections: List[Detection], frame: np.ndarray,
               ts_ms: float) -> List[TagTrack]:
        """Associate detections, return tracks that just CLOSED (finalised)."""
        ids = list(self._active.keys())
        # greedy IoU matching, highest IoU pairs first
        pairs = []
        for di, det in enumerate(detections):
            for tid in ids:
                iou = self._active[tid].last_bbox.iou(det.bbox)
                if iou >= self.iou_thr:
                    pairs.append((iou, di, tid))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for iou, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            a = self._active[tid]
            det = detections[di]
            a.last_bbox = det.bbox
            a.misses = 0
            a.hits += 1
            update_track(a.track, frame, det.bbox, det.score,
                         det.frame_idx, ts_ms, self.cfg)

        for di, det in enumerate(detections):
            if di not in used_d:
                self._new(det, frame, ts_ms)

        closed: List[TagTrack] = []
        for tid in ids:
            if tid in used_t:
                continue
            a = self._active[tid]
            a.misses += 1
            if a.misses > self.close_after:
                if a.hits >= self.min_hits:
                    closed.append(a.track)
                del self._active[tid]
        return closed

    def flush(self) -> List[TagTrack]:
        """End of video: finalise all still-open tracks meeting min_hits."""
        out = [a.track for a in self._active.values()
               if a.hits >= self.min_hits]
        self._active.clear()
        return out
