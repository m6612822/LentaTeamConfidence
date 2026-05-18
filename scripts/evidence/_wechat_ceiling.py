"""Decisive: WeChat QR (CNN detect + super-res) over the whole video.
Does it decode QRs that zxing (0/412) could not? Ceiling with WeChat."""
import csv, sys, time
sys.path.insert(0, "src")
import av, cv2

VIDEO, CSV = "data/25_12-20/25_12-20.mp4", "data/25_12-20/25_12-20.csv"
gt = list(csv.DictReader(open(CSV, encoding="utf-8")))
gt_qr = {r["qr_code_barcode"].strip() for r in gt
         if r["qr_code_barcode"].strip() not in ("", "нет")}
print(f"GT distinct qr barcodes: {len(gt_qr)}; sample: {list(gt_qr)[:3]}")

w = cv2.wechat_qrcode.WeChatQRCode(
    "models/wechat_qr/detect.prototxt", "models/wechat_qr/detect.caffemodel",
    "models/wechat_qr/sr.prototxt", "models/wechat_qr/sr.caffemodel")

c = av.open(VIDEO); vs = c.streams.video[0]
payloads, n, t0 = [], 0, time.time()
for i, fr in enumerate(c.decode(video=0)):
    if i % 3:
        continue
    n += 1
    img = fr.to_ndarray(format="bgr24")
    try:
        texts, _ = w.detectAndDecode(img)
    except Exception:
        texts = []
    for t in texts or []:
        if t:
            payloads.append(t)
print(f"scanned {n} frames in {time.time()-t0:.0f}s")
print(f"total QR decodes: {len(payloads)}; distinct: {len(set(payloads))}")
for p in list(set(payloads))[:5]:
    print("  QR:", p[:160])
# does any decoded payload contain a GT qr barcode?
hit = sum(1 for g in gt_qr if any(g in p for p in payloads))
print(f"GT qr barcodes appearing in some decoded payload: {hit}/{len(gt_qr)}")
