"""Throwaway: crop clean GT tag regions at a timestamp, make a montage."""
import csv
import av, cv2, numpy as np

VIDEO, CSV, TARGET_MS = "data/25_12-20/25_12-20.mp4", "data/25_12-20/25_12-20.csv", 7168

def fnum(x):
    try: return float(str(x).replace(",", "."))
    except: return None

rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r["frame_timestamp"] == str(TARGET_MS)]

c = av.open(VIDEO); vs = c.streams.video[0]
best, bdt = None, 1e9
for fr in c.decode(video=0):
    t = float(fr.pts * vs.time_base) * 1000.0
    if abs(t - TARGET_MS) < bdt: bdt, best = abs(t - TARGET_MS), fr
    if t > TARGET_MS + 800: break
img = best.to_ndarray(format="bgr24")

crops = []
for i, r in enumerate(rows):
    x0, y0, x1, y1 = fnum(r["x_min"]), fnum(r["y_min"]), fnum(r["x_max"]), fnum(r["y_max"])
    if None in (x0, y0, x1, y1): continue
    pad = 8
    cx0, cy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    cx1, cy1 = min(img.shape[1], int(x1) + pad), min(img.shape[0], int(y1) + pad)
    crop = img[cy0:cy1, cx0:cx1]
    if crop.size == 0: continue
    h, w = crop.shape[:2]
    print(f"#{i} barcode={r['barcode']} name={r['product_name'][:40]!r} "
          f"bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) crop={w}x{h}")
    cv2.imwrite(f"debug_crop_{i}.png", crop)
    # normalize to fixed height for montage
    th = 260
    crops.append(cv2.resize(crop, (max(1, int(w * th / h)), th)))

if crops:
    H = 260
    W = sum(c.shape[1] + 10 for c in crops)
    canvas = np.full((H, W, 3), 255, np.uint8)
    x = 0
    for c_ in crops:
        canvas[0:H, x:x + c_.shape[1]] = c_
        x += c_.shape[1] + 10
    cv2.imwrite("debug_montage.png", canvas)
    print(f"montage {W}x{H} of {len(crops)} tags")
