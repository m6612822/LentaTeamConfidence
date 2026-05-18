"""Throwaway: extract a real frame at a GT frame_timestamp, overlay GT bboxes.
Verifies (1) video orientation/resolution, (2) that GT ts+bbox align with tags.
"""
import csv, sys
import av, cv2, numpy as np

VIDEO = "data/25_12-20/25_12-20.mp4"
CSV = "data/25_12-20/25_12-20.csv"
TARGET_MS = 7168  # has ~9 tags in GT

def fnum(x):
    try: return float(str(x).replace(",", "."))
    except: return None

rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r["frame_timestamp"] == str(TARGET_MS)]
print(f"GT rows @ {TARGET_MS}ms: {len(rows)}")

container = av.open(VIDEO)
vs = container.streams.video[0]
print(f"video: {vs.width}x{vs.height} avg_fps={float(vs.average_rate):.3f} "
      f"frames={vs.frames} dur_s={float(vs.duration * vs.time_base):.1f}"
      if vs.duration else f"video: {vs.width}x{vs.height}")

best = None
best_dt = 1e9
for frame in container.decode(video=0):
    t_ms = float(frame.pts * vs.time_base) * 1000.0
    dt = abs(t_ms - TARGET_MS)
    if dt < best_dt:
        best_dt, best = dt, (t_ms, frame)
    if t_ms > TARGET_MS + 1000:
        break

t_ms, frame = best
img = frame.to_ndarray(format="bgr24")
print(f"picked frame @ {t_ms:.0f}ms (Δ{best_dt:.0f}ms), shape={img.shape}")

for r in rows:
    x0, y0, x1, y1 = (fnum(r["x_min"]), fnum(r["y_min"]),
                      fnum(r["x_max"]), fnum(r["y_max"]))
    if None in (x0, y0, x1, y1):
        continue
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 4)
    cv2.putText(img, r["barcode"][:6], (int(x0), int(y0) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

out = "debug_frame.png"
cv2.imwrite(out, img)
# also a downscaled view for quick visual
small = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
cv2.imwrite("debug_frame_small.png", small)
print(f"wrote {out} ({img.shape[1]}x{img.shape[0]}) + debug_frame_small.png")
