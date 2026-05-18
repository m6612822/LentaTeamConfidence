"""Throwaway data-inspection: correct CSV parsing (stdlib csv handles quotes)."""
import csv, glob, collections, os, sys

for path in sorted(glob.glob("data/**/*.csv", recursive=True)):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"\n### {path}: EMPTY"); continue
    cols = list(rows[0].keys())
    print(f"\n### {path}  rows={len(rows)} cols={len(cols)}")
    print("filename distinct:", sorted({r['filename'] for r in rows}))
    def dist(c, n=12):
        cnt = collections.Counter(r.get(c, '<MISSING>') for r in rows)
        return cnt.most_common(n)
    for c in ("color","code","special_symbols","frame_timestamp"):
        print(f"  {c}: {dist(c)}")
    print("  discount_amount sample:", sorted({r['discount_amount'] for r in rows})[:8])
    print("  print_datetime sample:", sorted({r['print_datetime'] for r in rows})[:6])
    print("  price_default sample:", [r['price_default'] for r in rows[:4]])
    print("  price_discount != нет:", sum(1 for r in rows if r['price_discount'] != 'нет'))
    # bbox numeric range (comma decimal)
    def f(x):
        try: return float(str(x).replace(',', '.'))
        except: return None
    xs=[f(r['x_max']) for r in rows if f(r['x_max']) is not None]
    ys=[f(r['y_max']) for r in rows if f(r['y_max']) is not None]
    if xs and ys:
        print(f"  bbox x_max[min..max]={min(xs):.0f}..{max(xs):.0f}  y_max[min..max]={min(ys):.0f}..{max(ys):.0f}")
    print("  QR barcode==visual barcode:",
          sum(1 for r in rows if r['qr_code_barcode']==r['barcode']), "/", len(rows))
    # one full example row
    if path.endswith("25_12-20.csv"):
        print("  --- example row 1 ---")
        for k,v in rows[0].items(): print(f"    {k} = {v!r}")
