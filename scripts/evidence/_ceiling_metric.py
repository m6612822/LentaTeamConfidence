"""Upper-bound metric: assume PERFECT OCR/detection (use GT visual values),
apply ONLY our QR-inference + нет/empty strategy, score vs GT. Answers:
is >=80% even reachable with our field strategy?  (instant, no video/OCR)"""
import csv, glob, os, sys
sys.path.insert(0, "src")
from lenta.config import load_config
from lenta.schema import QR_FIELDS

cfg = load_config()["qr_inference"]

def num(x):
    x = str(x).strip().replace(",", ".")
    try: return float(x)
    except: return None

def fmt(v):
    if v is None: return ""
    s = f"{v:.2f}".rstrip("0").rstrip("."); return s or "0"

rows = []
for c in sorted(glob.glob("data/*/*.csv")):
    if os.sep+"Unlabeled"+os.sep in c: continue
    rows += list(csv.DictReader(open(c, encoding="utf-8")))

# scored fields = all except meta/coords (mirror self_eval)
SKIP = {"filename","x_min","y_min","x_max","y_max","frame_timestamp"}
fields = [k for k in rows[0] if k not in SKIP]

def norm(f, v):
    s = "" if v is None else str(v).strip()
    sl = s.casefold()
    if sl in ("нет","no"): return "нет"
    if s == "": return ""
    n = num(s)
    if n is not None and f not in ("barcode","qr_code_barcode","id_sku",
                                   "action_code_qr","code","print_datetime"):
        return f"{n:.2f}"
    return sl

passed = 0
per = {f:[0,0] for f in fields}
for r in rows:
    # PREDICTED row assuming perfect visual OCR (= GT visual) + our QR rule
    pred = {f: r.get(f,"") for f in fields if f not in QR_FIELDS}
    pd, pc = num(r.get("price_default")), num(r.get("price_card"))
    bc = r.get("barcode","").strip()
    pred["qr_code_barcode"] = bc if bc and bc!="нет" else ""
    pred["price1_qr"] = fmt(pd) if pd is not None else ""
    pred["price4_qr"] = fmt(pc) if pc is not None else ""
    pred["price2_qr"] = fmt(round(pd*cfg["price2_ratio"],2)) if pd else ""
    pred["price3_qr"] = cfg["price3_const"]
    for f in cfg["constant_net"]: pred[f] = "нет"
    ok = 0
    for f in fields:
        good = norm(f, pred.get(f)) == norm(f, r.get(f))
        ok += good; per[f][0]+=good; per[f][1]+=1
    if ok/len(fields) >= 0.8: passed += 1

print(f"rows={len(rows)} scored_fields={len(fields)}")
print(f"=== CEILING: tags >=80% (perfect OCR + our QR rule): "
      f"{passed}/{len(rows)} = {passed/len(rows)*100:.1f}% ===")
print("per-field accuracy under perfect-OCR assumption (worst first):")
for f,(c,t) in sorted(per.items(), key=lambda kv: kv[1][0]/kv[1][1]):
    print(f"  {f:24s} {c/t*100:5.1f}%")
