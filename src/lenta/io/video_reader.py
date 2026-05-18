"""PyAV video reader with correct millisecond timestamps.

CRITICAL: the robot videos have a non-zero stream `start_time` (e.g. pts of
frame 0 = 47462 @ tb=1/57600 -> 824 ms). GT `frame_timestamp` is measured
from 0 at frame 0, so the timestamp MUST subtract `start_time`:

    ms = (pts - stream.start_time) * time_base * 1000

Using raw `pts*tb` (or idx/fps on this VFR footage) shifts every frame by
~824 ms and silently misaligns GT boxes. This bug was caught in Phase 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import av
import numpy as np


@dataclass
class Frame:
    idx: int          # decode order index (0-based)
    ts_ms: float      # milliseconds from video start (GT-compatible)
    image: np.ndarray  # HxWx3 BGR uint8


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        with av.open(path) as c:
            vs = c.streams.video[0]
            self.width = int(vs.width)
            self.height = int(vs.height)
            self.start_pts = int(vs.start_time or 0)
            self.time_base = vs.time_base
            self.avg_fps = float(vs.average_rate) if vs.average_rate else 0.0
            self.n_frames = int(vs.frames or 0)
            self.duration_ms = (
                float(vs.duration * vs.time_base) * 1000.0 if vs.duration else 0.0
            )

    def _ms(self, pts: Optional[int], idx: int) -> float:
        if pts is None:
            # VFR fallback only if pts missing; avg_fps still ignores start
            return (idx / self.avg_fps * 1000.0) if self.avg_fps else float(idx)
        return max(0.0, float((pts - self.start_pts) * self.time_base) * 1000.0)

    def frames(self) -> Iterator[Frame]:
        """Stream every decoded frame (memory-safe: one frame at a time)."""
        with av.open(self.path) as c:
            for idx, fr in enumerate(c.decode(video=0)):
                yield Frame(idx, self._ms(fr.pts, idx),
                            fr.to_ndarray(format="bgr24"))

    def frame_at_ms(self, target_ms: float,
                    window_ms: float = 400.0) -> Optional[Frame]:
        """Nearest frame to target_ms via SEEK (O(1), not full decode).

        Seeks to the keyframe at/before the target pts, then decodes forward
        only until the target window is passed. ~20x faster than scanning all
        4K frames when only a handful of timestamps are needed.
        """
        if self.time_base in (None, 0):
            return self._scan_at_ms(target_ms, window_ms)
        target_pts = self.start_pts + int(round(
            (target_ms / 1000.0) / float(self.time_base)))
        best: Optional[Frame] = None
        best_dt = float("inf")
        with av.open(self.path) as c:
            vs = c.streams.video[0]
            try:
                c.seek(max(0, target_pts), stream=vs,
                       backward=True, any_frame=False)
            except av.AVError:
                return self._scan_at_ms(target_ms, window_ms)
            for idx, fr in enumerate(c.decode(video=0)):
                ms = self._ms(fr.pts, idx)
                dt = abs(ms - target_ms)
                if dt < best_dt:
                    best_dt = dt
                    best = Frame(idx, ms, fr.to_ndarray(format="bgr24"))
                if ms > target_ms + window_ms and best is not None:
                    break
        return best

    def frames_window(self, t0_ms: float, t1_ms: float) -> Iterator[Frame]:
        """Stream frames whose ts is in [t0_ms, t1_ms] using SEEK (cheap for
        late timestamps; avoids decoding the whole video per tag)."""
        if self.time_base in (None, 0):
            for fr in self.frames():
                if fr.ts_ms > t1_ms:
                    break
                if fr.ts_ms >= t0_ms:
                    yield fr
            return
        seek_pts = self.start_pts + int(round(
            (max(0.0, t0_ms) / 1000.0) / float(self.time_base)))
        with av.open(self.path) as c:
            vs = c.streams.video[0]
            try:
                c.seek(max(0, seek_pts), stream=vs,
                       backward=True, any_frame=False)
            except av.AVError:
                return
            for idx, fr in enumerate(c.decode(video=0)):
                ms = self._ms(fr.pts, idx)
                if ms > t1_ms:
                    break
                if ms >= t0_ms:
                    yield Frame(idx, ms, fr.to_ndarray(format="bgr24"))

    def _scan_at_ms(self, target_ms: float, window_ms: float) -> Optional[Frame]:
        best, best_dt = None, float("inf")
        for fr in self.frames():
            dt = abs(fr.ts_ms - target_ms)
            if dt < best_dt:
                best_dt, best = dt, fr
            if fr.ts_ms > target_ms + window_ms and best is not None:
                break
        return best
