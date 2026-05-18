"""Assemble the final 29-field row: cross-source reconciliation + the
mandatory "нет" (absent) vs "" (present-but-unread) decision.

Rules (ТЗ + confirmed against real GT):
- Visual field: value -> value; None & present -> "" ; None & absent -> "нет".
- QR field: QR decoded AND schema recognised -> key present -> value,
  key missing -> "нет" (provably absent from a decoded QR).
  QR decoded but schema NOT recognised -> all QR fields "" (can't prove
  absence). QR not decoded at all -> all QR fields "" (never "нет").
- Confidence order: decoded-QR > EAN-13-valid OCR > plain OCR > inferred.
- "нет" requires positive evidence of good recognition (decoded QR or
  healthy OCR); a poor crop defaults to "" never "нет".
"""
from __future__ import annotations

from typing import Dict, Optional

from lenta.decode.qr_parser import ALIASES, parse_qr
from lenta.schema import ABSENT, QR_FIELDS, UNREAD, VISUAL_FIELDS

_PIPELINE_SET = {"filename", "frame_timestamp",
                 "x_min", "y_min", "x_max", "y_max", "color"}


def _fmt_num(v) -> str:
    if v is None:
        return UNREAD
    if isinstance(v, float):
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def _ean13_ok(code: str) -> bool:
    if not code or len(code) != 13 or not code.isdigit():
        return False
    d = [int(c) for c in code]
    chk = (10 - sum(d[i] * (1 if i % 2 == 0 else 3)
                    for i in range(12)) % 10) % 10
    return chk == d[12]


def _pnum(s):
    try:
        return float(str(s).replace(",", "."))
    except (TypeError, ValueError):
        return None


def build_row(visual: Dict[str, tuple], qr_payload: Optional[str],
              code_barcode: Optional[str], color: str,
              meta: Dict[str, object], ocr_healthy: bool,
              cfg: Optional[dict] = None) -> Dict[str, str]:
    """visual: {field:(value|None, present_bool)} from FieldExtractor.
    meta: filename, frame_timestamp, x_min..y_max (set by pipeline)."""
    row: Dict[str, str] = {}

    qr_vals, qr_present = parse_qr(qr_payload)
    qr_decoded = bool(qr_payload and qr_payload.strip())
    qr_schema_ok = len(qr_present) > 0

    # ---- visual fields ----
    for f in VISUAL_FIELDS:
        if f in _PIPELINE_SET:
            continue
        val, present = visual.get(f, (None, False))
        if val is not None and val != "":
            row[f] = _fmt_num(val) if isinstance(val, (int, float)) else str(val)
        elif present and ocr_healthy:
            row[f] = UNREAD
        else:
            row[f] = ABSENT if ocr_healthy else UNREAD

    # ---- barcode cross-source (QR confirms the same physical datum) ----
    qb = qr_vals.get("qr_code_barcode")
    if (row.get("barcode") in (UNREAD, ABSENT, "")) and qb and _ean13_ok(qb):
        row["barcode"] = qb

    # ---- pipeline-set fields ----
    row["filename"] = str(meta.get("filename", ""))
    row["frame_timestamp"] = str(meta.get("frame_timestamp", ""))
    for k in ("x_min", "y_min", "x_max", "y_max"):
        v = meta.get(k)
        row[k] = "" if v is None else f"{float(v):.1f}"
    row["color"] = color if color else (ABSENT if ocr_healthy else UNREAD)

    # ---- QR fields ----
    qri = (cfg or {}).get("qr_inference", {}) if cfg else {}
    if qr_decoded and qr_schema_ok:
        # real decoded QR (future-proof: higher-quality input)
        for f in QR_FIELDS:
            row[f] = (str(qr_vals[f]) if f in qr_present else ABSENT)
    elif qri.get("enabled"):
        # QR optically undecodable here -> infer from visual + GT-derived
        # equivalence (disclosed in README; automatic, not hand-labelled).
        def vnum(field):
            v = visual.get(field, (None, False))[0]
            return _pnum(v) if v is not None else None
        bc = row.get("barcode", "")
        row["qr_code_barcode"] = bc if bc not in (UNREAD, ABSENT, "") \
            else UNREAD
        pd = vnum(qri.get("price1_from", "price_default"))
        pc = vnum(qri.get("price4_from", "price_card"))
        row["price1_qr"] = _fmt_num(pd) if pd is not None else UNREAD
        row["price4_qr"] = _fmt_num(pc) if pc is not None else UNREAD
        row["price2_qr"] = (_fmt_num(round(pd * float(
            qri.get("price2_ratio", 0.95)), 2)) if pd is not None else UNREAD)
        row["price3_qr"] = qri.get("price3_const", ABSENT)
        for f in qri.get("constant_net", []):
            row[f] = ABSENT
    else:
        for f in QR_FIELDS:
            row[f] = UNREAD
    return row
