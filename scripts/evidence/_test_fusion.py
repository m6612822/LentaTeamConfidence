"""Validate fusion: single-frame OCR vs multi-frame fused OCR on the same
tags (incl. the partially-readable ABRAU LIGHT). Decisive go/no-go."""
import csv, glob, os, sys
sys.path.insert(0, "src")
import cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
from lenta.config import load_config
from lenta.preprocess.fusion import fuse_tag
from lenta.types import BBox

cfg = load_config()
eng = RapidOCR(rec_model_path="models/ocr_cyrillic/rec_cyrillic.onnx",
               rec_keys_path="models/ocr_cyrillic/cyrillic_dict.txt")

def fn(x):
    try: return float(str(x).replace(",", "."))
    except: return None

# pick a few large GT tags from 25_12-20 / 26_12-20
items = []
for cpath in ("data/25_12-20/25_12-20.csv", "data/26_12-20/26_12-20.csv"):
    zone = os.path.basename(os.path.dirname(cpath))
    mp4 = os.path.join(os.path.dirname(cpath), zone + ".mp4")
    for r in csv.DictReader(open(cpath, encoding="utf-8")):
        x0,y0,x1,y1 = fn(r["x_min"]),fn(r["y_min"]),fn(r["x_max"]),fn(r["y_max"])
        ts = fn(r["frame_timestamp"])
        if None in (x0,y0,x1,y1,ts): continue
        items.append(((x1-x0)*(y1-y0), mp4, ts, BBox(x0,y0,x1,y1), r))
items.sort(key=lambda t: t[0], reverse=True)

def best_ocr(img):
    best=""
    for op in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        res,_ = eng(cv2.rotate(img, op))
        j=" ".join(t for _,t,_ in (res or []))
        if len(j)>len(best): best=j
    return best

for area, mp4, ts, b, r in items[:6]:
    print(f"\n=== {os.path.basename(mp4)} ts={ts:.0f} area={area:.0f} "
          f"GT={r['product_name'][:38]!r} def={r['price_default']} "
          f"card={r['price_card']} ===")
    fr = __import__("lenta.io.video_reader", fromlist=["VideoReader"]) \
        .VideoReader(mp4).frame_at_ms(ts)
    if fr is not None:
        c = fr.image[int(b.y0):int(b.y1), int(b.x0):int(b.x1)]
        up = cv2.resize(c, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        print(f" single : {best_ocr(up)[:130]}")
    fused = fuse_tag(mp4, ts, b, cfg)
    if fused is not None:
        cv2.imwrite(f"fuse_{int(ts)}.png", fused)
        print(f" FUSED  ({fused.shape[1]}x{fused.shape[0]}): {best_ocr(fused)[:130]}")
    else:
        print(" FUSED  : (none)")
