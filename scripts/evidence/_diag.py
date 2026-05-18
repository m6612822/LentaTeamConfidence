"""Diagnostic: do GT boxes align with tags? Test ts=0 (frame 0, unambiguous)
and dump pts of first frames + check several timestamps."""
import csv, sys
import av, cv2, numpy as np

VIDEO, CSV = "data/25_12-20/25_12-20.mp4", "data/25_12-20/25_12-20.csv"

def fnum(x):
    try: return float(str(x).replace(",", "."))
    except: return None

allrows = list(csv.DictReader(open(CSV, encoding="utf-8")))
ts_vals = sorted({r["frame_timestamp"] for r in allrows}, key=lambda s: float(s))
print("distinct ts:", ts_vals[:12])

c = av.open(VIDEO); vs = c.streams.video[0]
print(f"video {vs.width}x{vs.height} tb={vs.time_base} avg_rate={vs.average_rate} "
      f"frames={vs.frames} start_time={vs.start_time}")

# dump first 5 frame pts -> ms
c2 = av.open(VIDEO); vs2 = c2.streams.video[0]
for i, fr in enumerate(c2.decode(video=0)):
    print(f"  frame{i} pts={fr.pts} ms={float(fr.pts*vs2.time_base)*1000:.1f}")
    if i >= 4: break

for TS in ["0", ts_vals[1] if len(ts_vals) > 1 else "0"]:
    rows = [r for r in allrows if r["frame_timestamp"] == TS]
    cc = av.open(VIDEO); vss = cc.streams.video[0]
    best, bdt = None, 1e9
    for fr in cc.decode(video=0):
        t = float(fr.pts * vss.time_base) * 1000.0
        if abs(t - float(TS)) < bdt: bdt, best = abs(t - float(TS)), (t, fr)
        if t > float(TS) + 600: break
    t_ms, fr = best
    img = fr.to_ndarray(format="bgr24")
    for r in rows:
        x0,y0,x1,y1 = fnum(r["x_min"]),fnum(r["y_min"]),fnum(r["x_max"]),fnum(r["y_max"])
        if None in (x0,y0,x1,y1): continue
        cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),(0,0,255),5)
    half = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    fn = f"diag_ts{TS}.png"
    cv2.imwrite(fn, half)
    print(f"{fn}: ts={TS} picked={t_ms:.0f}ms rows={len(rows)} (img {img.shape[1]}x{img.shape[0]})")
