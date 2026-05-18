"""Unit tests for the data-independent core: QR parsing, dedup, CSV, EAN-13.
Run: PYTHONPATH=src .venv/bin/python -m pytest -q
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lenta.aggregate.dedup import deduplicate
from lenta.assemble.fields import _ean13_ok
from lenta.config import load_config
from lenta.decode.qr_parser import parse_qr
from lenta.io.csv_writer import write_csv
from lenta.schema import CSV_COLUMNS
from lenta.types import BBox, TagTrack

CFG = load_config()


# ---- QR parser: full keys + aliases, 3 strategies ----
def test_qr_json_full_keys():
    v, p = parse_qr('{"barcode":"4670025474665","price1":252.63,'
                    '"actionCode":"AC9"}')
    assert v["qr_code_barcode"] == "4670025474665"
    assert v["price1_qr"] == "252.63"
    assert v["action_code_qr"] == "AC9"
    assert "qr_code_barcode" in p


def test_qr_aliases_and_querystring():
    v, p = parse_qr("b=4670025474665;p1=252,63;wL1C=6;aP=99.9")
    assert v["qr_code_barcode"] == "4670025474665"
    assert v["price1_qr"] == "252.63"        # comma -> dot
    assert v["wholesale_level_1_count"] == "6"
    assert v["action_price_qr"] == "99.9"


def test_qr_empty():
    v, p = parse_qr("")
    assert v == {} and p == set()
    v, p = parse_qr(None)
    assert v == {} and p == set()


# ---- EAN-13 checksum ----
def test_ean13():
    assert _ean13_ok("4670025474665")
    assert not _ean13_ok("4670025474664")
    assert not _ean13_ok("123")


# ---- dedup: barcode primary key collapses tracks ----
def _track(tid, bc, ts, x, q):
    t = TagTrack(track_id=tid)
    t.barcode = bc
    t.best_ts_ms = ts
    t.best_bbox = BBox(x, 0, x + 50, 50)
    t.best_quality = q
    return t


def test_dedup_barcode_primary():
    a = _track(1, "4670025474665", 100, 10, 0.5)
    b = _track(2, "4670025474665", 5000, 900, 0.9)  # same barcode -> merge
    c = _track(3, "4670025474658", 200, 20, 0.7)
    reps = deduplicate([a, b, c], CFG)
    assert len(reps) == 2
    bc_rep = [r for r in reps if r.barcode == "4670025474665"][0]
    assert bc_rep.track_id == 2  # higher quality wins


def test_dedup_spatial_temporal_when_no_barcode():
    a = _track(1, None, 1000, 100, 0.6)
    b = _track(2, None, 1200, 110, 0.8)   # close in ts+space -> same tag
    c = _track(3, None, 1100, 900, 0.7)   # far in space -> different tag
    reps = deduplicate([a, b, c], CFG)
    assert len(reps) == 2


# ---- CSV writer: schema lock + нет/empty preserved ----
def test_csv_writer_schema_and_quoting():
    rows = [{"filename": "v.mp4", "product_name": "Вино, 0,75",
             "barcode": "4670025474665", "price_default": 252.63,
             "code": "нет", "color": "red"}]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "o.csv")
        n = write_csv(rows, p)
        assert n == 1
        head, row = open(p, encoding="utf-8").read().splitlines()[:2]
    assert head.split(",") == CSV_COLUMNS
    assert '"Вино, 0,75"' in row     # comma field quoted
    assert "нет" in row              # absent marker preserved
