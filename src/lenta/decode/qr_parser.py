"""Tolerant QR payload parser -> canonical CSV fields.

The QR schema is unknown until we see real data, but the field/alias map
is fixed by ТЗ. Strategy (first success wins):
  1. JSON (with light cleanup of single quotes / trailing commas)
  2. delimited key=value / key:value  (split on ; & newline, then = :)
  3. regex scan for any known alias token followed by a value

Returns (values, present_keys):
  values        canonical_field -> normalized string value
  present_keys  set of canonical fields explicitly present in the payload
                (used by the "нет" vs empty decision: a key absent from a
                 SUCCESSFULLY decoded QR is provably absent -> "нет")
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional, Set, Tuple

# canonical CSV field -> accepted QR keys (full + alias), case-insensitive
ALIASES: Dict[str, Tuple[str, ...]] = {
    "qr_code_barcode": ("barcode", "b"),
    "price1_qr": ("price1", "p1"),
    "price2_qr": ("price2", "p2"),
    "price3_qr": ("price3", "p3"),
    "price4_qr": ("price4", "p4"),
    "wholesale_level_1_count": ("wholesalelevel1count", "wl1c"),
    "wholesale_level_1_price": ("wholesalelevel1price", "wl1p"),
    "wholesale_level_2_count": ("wholesalelevel2count", "wl2c"),
    "wholesale_level_2_price": ("wholesalelevel2price", "wl2p"),
    "action_price_qr": ("actionprice", "ap"),
    "action_code_qr": ("actioncode", "ac"),
}
_KEY_TO_FIELD = {a: f for f, al in ALIASES.items() for a in al}
_ALL_ALIAS_RE = re.compile(
    r"(?P<k>" + "|".join(sorted(_KEY_TO_FIELD, key=len, reverse=True)) + r")"
    r"\s*[:=]\s*(?P<v>[^;&,\n]+)",
    re.IGNORECASE,
)


def _norm_value(field: str, raw: str) -> str:
    raw = str(raw).strip().strip("\"'")
    if field == "action_code_qr":
        return raw  # codes are not numeric
    # numeric-ish fields: unify decimal separator, drop currency/spaces
    m = re.search(r"-?\d[\d\s.,]*", raw)
    if not m:
        return raw
    num = m.group(0).replace(" ", "").replace(",", ".")
    if num.count(".") > 1:  # thousands dots: keep last as decimal
        head, _, tail = num.rpartition(".")
        num = head.replace(".", "") + "." + tail
    return num


def _ingest(d: dict, values: Dict[str, str], present: Set[str]) -> None:
    for k, v in d.items():
        field = _KEY_TO_FIELD.get(str(k).strip().lower())
        if field and field not in values:
            values[field] = _norm_value(field, v)
            present.add(field)


def parse_qr(payload: Optional[str]) -> Tuple[Dict[str, str], Set[str]]:
    values: Dict[str, str] = {}
    present: Set[str] = set()
    if not payload or not payload.strip():
        return values, present
    text = payload.strip()

    # 1) JSON
    for candidate in (text, re.sub(r",\s*([}\]])", r"\1", text.replace("'", '"'))):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                _ingest(obj, values, present)
                if values:
                    return values, present
        except (ValueError, TypeError):
            pass

    # 2) delimited key=value / key:value
    pairs = {}
    for tok in re.split(r"[;&\n]+", text):
        m = re.match(r"\s*([\w]+)\s*[:=]\s*(.+)", tok)
        if m:
            pairs[m.group(1)] = m.group(2)
    if pairs:
        _ingest(pairs, values, present)
        if values:
            return values, present

    # 3) regex fallback
    for m in _ALL_ALIAS_RE.finditer(text):
        field = _KEY_TO_FIELD[m.group("k").lower()]
        if field not in values:
            values[field] = _norm_value(field, m.group("v"))
            present.add(field)
    return values, present
