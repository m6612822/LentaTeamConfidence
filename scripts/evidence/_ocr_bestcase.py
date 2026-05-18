"""Realistic OCR ceiling: take the LARGEST GT boxes (closest robot pass),
extract the correct frame (start_time-fixed), enhance, OCR with Cyrillic
model. Shows best-case OCR quality with perfect detection+best-frame."""
import csv, glob, os, sys
sys.path.insert(0, "src")
import av, cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR

def fn(x):
    try: return float(str(x).replace(",", "."))
    except: return None

# gather all GT rows with area, keep top-N largest
items = []
for cpath in sorted(glob.glob("data/*/*.csv")):
    if os.sep+"Unlabeled"+os.sep in cpath: continue
    zone = os.path.basename(os.path.dirname(cpath))
    mp4 = os.path.join(os.path.dirname(cpath), zone + ".mp4")
    for r in csv.DictReader(open(cpath, encoding="utf-8")):
        x0,y0,x1,y1 = fn(r["x_min"]),fn(r["y_min"]),fn(r["x_max"]),fn(r["y_max"])
        ts = fn(r["frame_timestamp"])
        if None in (x0,y0,x1,y1,ts): continue
        items.append(((x1-x0)*(y1-y0), mp4, ts, (x0,y0,x1,y1), r))
items.sort(key=lambda t: t[0], reverse=True)
top = items[:10]

eng = RapidOCR(rec_model_path="models/ocr_cyrillic/rec_cyrillic.onnx",
               rec_keys_path="models/ocr_cyrillic/cyrillic_dict.txt")

def enhance(crop, tgt=1100):
    h,w = crop.shape[:2]; s = tgt/max(h,w)
    if s>1: crop = cv2.resize(crop,(int(w*s),int(h*s)),interpolation=cv2.INTER_CUBIC)
    d = cv2.fastNlMeansDenoisingColored(crop,None,3,3,7,21)
    return cv2.addWeighted(d,1.5,cv2.GaussianBlur(d,(0,0),3),-0.5,0)

cache = {}
for area, mp4, ts, (x0,y0,x1,y1), r in top:
    if mp4 not in cache:
        c = av.open(mp4); vs = c.streams.video[0]; cache[mp4] = (c, vs, vs.start_time or 0)
    c, vs, st = cache[mp4]
    tgt = st + int(round((ts/1000.0)/float(vs.time_base)))
    c.seek(max(0,tgt), stream=vs, backward=True, any_frame=False)
    best,bdt=None,9e9
    for fr in c.decode(video=0):
        ms=float((fr.pts-st)*vs.time_base)*1000
        if abs(ms-ts)<bdt: bdt,best=abs(ms-ts),fr
        if ms>ts+400 and best is not None: break
    img=best.to_ndarray(format="bgr24")
    crop=img[int(y0):int(y1), int(x0):int(x1)]
    print(f"\narea={area:.0f} {os.path.basename(mp4)} crop={crop.shape[1]}x{crop.shape[0]}")
    print(f"  GT name={r['product_name'][:45]!r} def={r['price_default']} card={r['price_card']} bc={r['barcode']}")
    txts=[]
    for deg,op in [(90,cv2.ROTATE_90_CLOCKWISE),(270,cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        res,_=eng(enhance(cv2.rotate(crop,op)))
        joined=" ".join(t for _,t,_ in (res or []))
        txts.append((deg,joined))
        print(f"  OCR rot{deg}: {joined[:140]}")
