"""Decisive viability test: scan ALL frames of a video, decode every
barcode/QR anywhere in-frame, compare the union to the GT barcodes.
Answers: with ideal best-frame selection, what fraction of tags' codes
are decodable at all in this video?"""
import csv, sys, time
sys.path.insert(0, "src")
import av, cv2, numpy as np, zxingcpp

VIDEO, CSV = "data/25_12-20/25_12-20.mp4", "data/25_12-20/25_12-20.csv"
gt = list(csv.DictReader(open(CSV, encoding="utf-8")))
gt_bc = {r["barcode"].strip() for r in gt
         if r["barcode"].strip() and r["barcode"].strip() != "нет"}
gt_qr = {r["qr_code_barcode"].strip() for r in gt
         if r["qr_code_barcode"].strip() and r["qr_code_barcode"].strip() != "нет"}
print(f"GT: {len(gt)} rows, {len(gt_bc)} distinct barcodes, "
      f"{len(gt_qr)} distinct qr barcodes")

found_bc, found_qr, qr_payloads = set(), set(), []
c = av.open(VIDEO); vs = c.streams.video[0]
t0 = time.time(); n = 0
for i, fr in enumerate(c.decode(video=0)):
    if i % 2:                       # every 2nd frame
        continue
    n += 1
    img = fr.to_ndarray(format="bgr24")
    for im in (img, cv2.resize(img, None, fx=0.5, fy=0.5)):
        for r in zxingcpp.read_barcodes(im):
            fmt = str(r.format).split(".")[-1]
            if fmt == "QRCode":
                found_qr.add(r.text)
                qr_payloads.append(r.text)
            else:
                found_bc.add(r.text)
print(f"scanned {n} frames in {time.time()-t0:.0f}s")
print(f"decoded distinct 1D barcodes: {len(found_bc)}")
print(f"  of GT barcodes covered: {len(found_bc & gt_bc)}/{len(gt_bc)} "
      f"= {len(found_bc & gt_bc)/max(1,len(gt_bc))*100:.0f}%")
print(f"decoded distinct QR: {len(found_qr)}")
print(f"  of GT qr barcodes covered: "
      f"{len(found_qr & gt_qr) if gt_qr else 0}/{len(gt_qr)}")
if qr_payloads:
    print("sample QR payload:", qr_payloads[0][:200])
