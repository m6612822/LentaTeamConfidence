"""Deduplicate tracks -> one row per unique physical price tag.

Priority per ТЗ:
  1. decoded BARCODE = primary key (exact; collapses all frames/tracks of a
     tag regardless of tracking errors).
  2. spatial-temporal key (frame_timestamp + bbox) with tolerances, for tags
     with no decoded barcode.
A track with no usable key at all (no valid bbox) is dropped (matched=False).
For a merged group the representative is the track with the best-quality
frame (the moment of final determination).
"""
from __future__ import annotations

from typing import Dict, List

from lenta.types import TagTrack


def _norm_barcode(bc) -> str:
    if not bc:
        return ""
    s = str(bc).strip()
    return "" if s in ("", "нет") else s


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def deduplicate(tracks: List[TagTrack], cfg: dict) -> List[TagTrack]:
    d = cfg["dedup"]
    ts_tol = float(d["ts_tolerance_ms"])
    ctr_tol = float(d["bbox_center_tol_px"])

    valid = [t for t in tracks if t.best_bbox is not None]
    for t in valid:
        t.matched = True

    groups: Dict[str, List[TagTrack]] = {}
    no_bc: List[TagTrack] = []
    for t in valid:
        bc = _norm_barcode(t.barcode)
        if bc:
            groups.setdefault("bc:" + bc, []).append(t)
        else:
            no_bc.append(t)

    # Pass 2: spatial-temporal union-find over barcode-less tracks
    uf = _UF(len(no_bc))
    for i in range(len(no_bc)):
        ci = no_bc[i].best_bbox.center
        for j in range(i + 1, len(no_bc)):
            if abs(no_bc[i].best_ts_ms - no_bc[j].best_ts_ms) > ts_tol:
                continue
            cj = no_bc[j].best_bbox.center
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist <= ctr_tol:
                uf.union(i, j)
    st_groups: Dict[int, List[TagTrack]] = {}
    for idx, t in enumerate(no_bc):
        st_groups.setdefault(uf.find(idx), []).append(t)
    for k, members in st_groups.items():
        groups[f"st:{k}"] = members

    reps: List[TagTrack] = []
    for members in groups.values():
        rep = max(members, key=lambda t: t.best_quality)
        reps.append(rep)

    reps.sort(key=lambda t: (t.best_ts_ms,
                             t.best_bbox.x0 if t.best_bbox else 0.0))
    return reps
