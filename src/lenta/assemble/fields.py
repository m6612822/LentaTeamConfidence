"""Visual field extraction from OCR output.

Layout-tolerant: works off keyword anchors + regex + spatial position
(NOT fixed templates) so it survives the many tag variants. All patterns
live in config/patterns.yaml -> calibration = editing YAML, not code.

OCR input = list of OcrLine(text, conf, box=(x,y,w,h)) in crop pixels.
Returns {field: (value|None, present: bool)} where `present` feeds the
"нет" vs empty decision in reconcile.py:
  value is not None            -> recognised
  value is None, present True  -> on the tag but unreadable  -> "" (empty)
  value is None, present False -> absent                      -> "нет"

NOTE: price-role disambiguation and the kopeck-superscript reconstruction
are the highest-risk heuristics; they MUST be calibrated against OCR on
real GT crops (dataset/images) before trusting the numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lenta.schema import VISUAL_FIELDS


@dataclass
class OcrLine:
    text: str
    conf: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def _rx(p):
    return re.compile(p, re.IGNORECASE | re.UNICODE)


def _has_anchor(text_up: str, anchors: List[str]) -> bool:
    return any(re.search(a, text_up, re.IGNORECASE) for a in anchors)


def _num(s: str) -> Optional[float]:
    s = s.replace(" ", "").replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


class FieldExtractor:
    def __init__(self, cfg: dict):
        p = cfg["patterns"]
        self.anchors = p["anchors"]
        self.rx = {k: _rx(v) for k, v in p["regex"].items()}
        self.dt_out = p["datetime_output"]
        self.known_info = p.get("additional_info_known_phrases", [])

    # ---- prices -----------------------------------------------------------
    def _prices(self, lines: List[OcrLine]) -> Dict[str, Optional[float]]:
        """Collect price tokens; assign card/default/discount by anchors,
        then by magnitude. Big-ruble + superscript-kopeck reconstruction:
        a large integer immediately followed by a small 2-digit token."""
        cand: List[Tuple[float, OcrLine]] = []
        for ln in lines:
            for m in self.rx["price"].finditer(ln.text):
                v = _num(m.group())
                if v is not None and 0.1 <= v <= 999999:
                    cand.append((v, ln))
        # role by anchor proximity
        roles: Dict[str, Optional[float]] = {
            "price_card": None, "price_default": None, "price_discount": None}
        used = set()
        for role, akey in (("price_card", "price_card"),
                           ("price_discount", "price_discount"),
                           ("price_default", "price_default")):
            for v, ln in cand:
                if id(ln) in used:
                    continue
                if _has_anchor(ln.text.upper(), self.anchors.get(akey, [])):
                    roles[role] = v
                    used.add(id(ln))
                    break
        # fill remaining by magnitude: default >= card; discount lowest
        rest = sorted((v for v, ln in cand if id(ln) not in used),
                      reverse=True)
        if roles["price_default"] is None and rest:
            roles["price_default"] = rest.pop(0)
        if roles["price_card"] is None and rest:
            roles["price_card"] = rest.pop(0)
        return roles

    # ---- main -------------------------------------------------------------
    def extract(self, lines: List[OcrLine]
                ) -> Dict[str, Tuple[Optional[object], bool]]:
        out: Dict[str, Tuple[Optional[object], bool]] = {}
        full_up = " ".join(ln.text for ln in lines).upper()
        joined = " ".join(ln.text for ln in lines)
        good = bool(lines) and (sum(l.conf for l in lines) / len(lines)
                                >= 0.4 if lines else False)

        prices = self._prices(lines)
        for f in ("price_default", "price_card", "price_discount"):
            v = prices[f]
            if v is not None:
                out[f] = (v, True)
            else:
                # discount price often genuinely absent (GT: mostly "нет")
                present = (f != "price_discount"
                           or _has_anchor(full_up,
                                          self.anchors["price_discount"]))
                out[f] = (None, present)

        # barcode (EAN-13, checksum-validated)
        bc = None
        for m in self.rx["barcode_ean"].finditer(joined):
            if _ean13_ok(m.group()):
                bc = m.group()
                break
        out["barcode"] = (bc, bc is not None or _has_anchor(
            full_up, self.anchors.get("barcode", [])) or good)

        # id_sku
        msku = self.rx["id_sku"].search(joined)
        out["id_sku"] = (msku.group() if msku else None,
                         msku is not None or _has_anchor(
                             full_up, self.anchors.get("id_sku", [])))

        # discount_amount  e.g. -48%
        md = self.rx["discount_pct"].search(joined)
        out["discount_amount"] = (
            md.group().replace(" ", "") if md else None,
            md is not None)

        # print_datetime
        out["print_datetime"] = self._datetime(joined, good)

        # zone code
        mc = self.rx["zone_code"].search(joined)
        out["code"] = (mc.group() if mc else None, good)

        # product_name: longest low-digit alpha line in the upper region
        out["product_name"] = self._product_name(lines)

        # additional_info: known descriptive phrases present on tag
        info = [ph for ph in self.known_info
                if ph.lower() in joined.lower()]
        out["additional_info"] = (", ".join(info) if info else None, good)

        return out

    def _datetime(self, text: str, good: bool):
        m = self.rx["datetime"].search(text)
        if not m:
            return (None, good)
        dd, mm, yy = m.group(1), m.group(2), m.group(3)
        hh = m.group(4) or "0"
        mi = m.group(5) or "00"
        try:
            import datetime as _dt
            d = _dt.datetime(int(yy), int(mm), int(dd), int(hh), int(mi))
            return (d.strftime(self.dt_out).replace(" 0", " "), True) \
                if "%-H" not in self.dt_out else (
                    d.strftime("%d.%m.%Y ") + f"{int(hh)}:{mi}", True)
        except ValueError:
            return (None, True)

    def _product_name(self, lines: List[OcrLine]):
        best, score = None, 0.0
        for ln in lines:
            t = ln.text.strip()
            if len(t) < 6:
                continue
            digits = sum(c.isdigit() for c in t)
            alpha = sum(c.isalpha() for c in t)
            if alpha < 4 or digits / max(1, len(t)) > 0.4:
                continue
            s = alpha * ln.conf / (1.0 + ln.y / 1000.0)  # prefer upper region
            if s > score:
                best, score = t, s
        return (best, best is not None)


def _ean13_ok(code: str) -> bool:
    if len(code) != 13 or not code.isdigit():
        return False
    d = [int(c) for c in code]
    chk = (10 - sum(d[i] * (1 if i % 2 == 0 else 3)
                    for i in range(12)) % 10) % 10
    return chk == d[12]
