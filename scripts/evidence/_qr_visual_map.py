"""Derive the QR<->visual field mapping from GT statistics (automatic data
analysis, NOT manual annotation). QR is optically undecodable in this
footage, so QR fields must be inferred from the visually-read data IF a
consistent equivalence exists. Quantify it."""
import csv, glob, os, collections

def num(x):
    x = str(x).strip().replace(",", ".")
    try: return round(float(x), 2)
    except: return None

rows = []
for c in sorted(glob.glob("data/*/*.csv")):
    if os.sep + "Unlabeled" + os.sep in c: continue
    rows += list(csv.DictReader(open(c, encoding="utf-8")))
print(f"total GT rows: {len(rows)}")

def eq_rate(a, b, numeric=False):
    n = ok = 0
    for r in rows:
        va, vb = r.get(a, ""), r.get(b, "")
        if numeric:
            na, nb = num(va), num(vb)
            if na is None or nb is None: continue
            n += 1; ok += (abs(na - nb) <= 0.01)
        else:
            va, vb = va.strip(), vb.strip()
            if va in ("", "нет") or vb in ("", "нет"): continue
            n += 1; ok += (va == vb)
    return ok, n

print("\n== identity checks (ok/comparable) ==")
print("qr_code_barcode == barcode:", eq_rate("qr_code_barcode", "barcode"))
for q in ("price1_qr", "price2_qr", "price3_qr", "price4_qr",
          "action_price_qr"):
    for v in ("price_default", "price_card", "price_discount"):
        ok, n = eq_rate(q, v, numeric=True)
        if n and ok / n > 0.5:
            print(f"{q} ~= {v}: {ok}/{n} = {ok/max(1,n)*100:.0f}%")

# value-set distribution of QR fields (how often 'нет' vs value)
print("\n== QR field fill stats ==")
for q in ("qr_code_barcode","price1_qr","price2_qr","price3_qr","price4_qr",
          "wholesale_level_1_count","wholesale_level_1_price",
          "wholesale_level_2_count","wholesale_level_2_price",
          "action_price_qr","action_code_qr"):
    c = collections.Counter(
        "нет" if r.get(q,"").strip()=="нет"
        else ("empty" if r.get(q,"").strip()=="" else "value") for r in rows)
    print(f"  {q:26s} {dict(c)}")

# what is price2_qr? compare to price_default*? and check ordering
print("\n== sample rows (visual vs QR prices) ==")
for r in rows[:6]:
    print(f"  def={r['price_default']:>9} card={r['price_card']:>9} "
          f"disc={r['price_discount']:>5} | p1={r['price1_qr']:>8} "
          f"p2={r['price2_qr']:>8} p3={r['price3_qr']:>5} "
          f"p4={r['price4_qr']:>8} aP={r['action_price_qr']:>5}")
