"""Honest delta: fused crop OCR WITHOUT vs WITH Real-ESRGAN ×4.
Same tags, same OCR — isolates the neural-SR contribution."""
import csv, os, sys, time
sys.path.insert(0, "src")
import cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
from lenta.config import load_config
from lenta.preprocess.fusion import _track, _align_fuse
from lenta.preprocess.superres import SuperRes
from lenta.io.video_reader import VideoReader
from lenta.types import BBox

cfg = load_config()
sr = SuperRes(cfg)
print("SuperRes enabled:", sr.enabled)
eng = RapidOCR(rec_model_path="models/ocr_cyrillic/rec_cyrillic.onnx",
               rec_keys_path="models/ocr_cyrillic/cyrillic_dict.txt")

def fnum(x):
    try: return float(str(x).replace(",", "."))
    except: return None

def ocr(im):
    best = ""
    for op in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        r, _ = eng(cv2.rotate(im, op))
        j = " ".join(t for _, t, _ in (r or []))
        if len(j) > len(best):
            best = j
    return best

items = []
for cp in ["data/25_12-20/25_12-20.csv", "data/26_12-20/26_12-20.csv"]:
    z = os.path.basename(os.path.dirname(cp)); mp4 = cp[:-4] + ".mp4"
    mp4 = os.path.join(os.path.dirname(cp), z + ".mp4")
    for r in csv.DictReader(open(cp, encoding="utf-8")):
        b = [fnum(r[k]) for k in ("x_min","y_min","x_max","y_max")]
        ts = fnum(r["frame_timestamp"])
        if None in b or ts is None: continue
        items.append(((b[2]-b[0])*(b[3]-b[1]), mp4, ts, BBox(*b), r))
items.sort(key=lambda t: t[0], reverse=True)

for area, mp4, ts, b, r in items[:5]:
    vr = VideoReader(mp4)
    crops = _track(vr, ts, b, cfg["fusion"]["window_ms"],
                    cfg["fusion"]["max_obs"], cfg["fusion"]["frame_step"])
    if not crops:
        print(f"ts={ts:.0f}: no track"); continue
    fused_native = _align_fuse(crops, 1.0)
    cubic = _align_fuse(crops, 2.5)
    t0 = time.time(); up = sr.upscale(fused_native); dt = time.time() - t0
    print(f"\nts={ts:.0f} GT={r['product_name'][:34]!r} "
          f"def={r['price_default']} card={r['price_card']}")
    print(f"  cubic2.5x : {ocr(cubic)[:120]}")
    print(f"  realesrgan({up.shape[1]}x{up.shape[0]}, {dt:.0f}s): "
          f"{ocr(up)[:120]}")
