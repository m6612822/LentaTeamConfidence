"""Shared lightweight data structures (no heavy deps -> easy to unit-test)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    def clip(self, w: float, h: float) -> "BBox":
        return BBox(max(0.0, min(self.x0, w)), max(0.0, min(self.y0, h)),
                    max(0.0, min(self.x1, w)), max(0.0, min(self.y1, h)))

    def iou(self, o: "BBox") -> float:
        ix0, iy0 = max(self.x0, o.x0), max(self.y0, o.y0)
        ix1, iy1 = min(self.x1, o.x1), min(self.y1, o.y1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        union = self.area + o.area - inter
        return inter / union if union > 0 else 0.0

    def touches_border(self, w: float, h: float, m: float = 2.0) -> bool:
        return (self.x0 <= m or self.y0 <= m
                or self.x1 >= w - m or self.y1 >= h - m)


@dataclass
class Detection:
    bbox: BBox
    score: float
    frame_idx: int
    ts_ms: float


@dataclass
class Observation:
    frame_idx: int
    ts_ms: float
    bbox: BBox
    quality: float          # combined sharpness/score (higher = better)


@dataclass
class TagTrack:
    track_id: int
    observations: List[Observation] = field(default_factory=list)
    # best frame (moment of FINAL determination, not first appearance)
    best_ts_ms: float = 0.0
    best_bbox: Optional[BBox] = None
    best_quality: float = -1.0
    # bounded set of top-K crops for decode retry (barcode = golden key)
    candidate_crops: List = field(default_factory=list)  # (quality, ndarray)
    # filled downstream
    barcode: Optional[str] = None
    fields: dict = field(default_factory=dict)
    matched: bool = False
