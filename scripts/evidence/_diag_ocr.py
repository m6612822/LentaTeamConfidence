"""Diagnose OCR yield: crop GT tags from a clear video, OCR at 0/90/180/270."""
import csv, sys, os
sys.path.insert(0, "src")
import av, cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR

VIDEO, CSV, TS = "data/25_12-20/25_12-20.mp4", "data/25_12-20/25_12-20.csv", 7168
def fn(x):
    try: return float(str(x).replace(",", "."))
    except: return None
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r["frame_timestamp"] == str(TS)][:4]

# seek frame
c = av.open(VIDEO); vs = c.streams.video[0]; st = vs.start_time or 0
tgt = st + int(round((TS/1000.0)/float(vs.time_base)))
c.seek(max(0,tgt), stream=vs, backward=True, any_frame=False)
best,bdt=None,9e9
for fr in c.decode(video=0):
    ms=float((fr.pts-st)*vs.time_base)*1000
    if abs(ms-TS)<bdt: bdt,best=abs(ms-TS),fr
    if ms>TS+400 and best is not None: break
img=best.to_ndarray(format="bgr24")
ocr=RapidOCR()
for i,r in enumerate(rows):
    x0,y0,x1,y1=fn(r["x_min"]),fn(r["y_min"]),fn(r["x_max"]),fn(r["y_max"])
    crop=img[int(y0)-6:int(y1)+6, int(x0)-6:int(x1)+6]
    cv2.imwrite(f"diag_ocr_{i}.png", crop)
    print(f"\n#{i} GT name={r['product_name'][:35]!r} price_def={r['price_default']} "
          f"bc={r['barcode']} crop={crop.shape[1]}x{crop.shape[0]}")
    for deg,op in [(0,None),(90,cv2.ROTATE_90_CLOCKWISE),
                   (180,cv2.ROTATE_180),(270,cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        im = crop if op is None else cv2.rotate(crop, op)
        res,_ = ocr(im)
        txt = " | ".join(f"{t}({c:.2f})" for _,t,c in (res or []))
        print(f"  rot{deg:3d}: {txt[:120] if txt else '(nothing)'}")
