"""Confirm start_time fix: GT ts=7168 with corrected ms = (pts-start)*tb*1000."""
import csv
import av, cv2

VIDEO, CSV, TS = "data/25_12-20/25_12-20.mp4", "data/25_12-20/25_12-20.csv", 7168
def fn(x):
    try: return float(str(x).replace(",", "."))
    except: return None
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r["frame_timestamp"] == str(TS)]
c = av.open(VIDEO); vs = c.streams.video[0]
st = vs.start_time or 0
best, bdt = None, 1e9
for fr in c.decode(video=0):
    ms = float((fr.pts - st) * vs.time_base) * 1000.0   # <-- start_time fix
    if abs(ms - TS) < bdt: bdt, best = abs(ms - TS), (ms, fr)
    if ms > TS + 400: break
ms, fr = best
img = fr.to_ndarray(format="bgr24")
for r in rows:
    x0,y0,x1,y1 = fn(r["x_min"]),fn(r["y_min"]),fn(r["x_max"]),fn(r["y_max"])
    cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),(0,0,255),5)
cv2.imwrite("diag_fix_7168.png", cv2.resize(img,(img.shape[1]//2,img.shape[0]//2)))
print(f"ts={TS} picked={ms:.0f}ms (Δ{bdt:.0f}) rows={len(rows)} -> diag_fix_7168.png")
