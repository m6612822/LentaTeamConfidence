"""Self-evaluation against the provided reference CSVs (eval ONLY — never
read before predictions exist; no rule branches on its content -> ТЗ-OK).

Headline metric (ТЗ): fraction of GT tags whose field accuracy >= 80%.
Plus a per-field error table to target field-logic tuning.

Matching pred<->GT: by barcode when present, else by (frame_timestamp
within tol AND bbox-center within tol). filename is ignored for matching
(GT filename is inconsistent: "25_12-20/2.mp4" vs "25_12-20").

  python scripts/self_eval.py --pred out.csv --data data [--ts-tol 1500]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

THRESH = 0.80
PRICE_FIELDS = {"price_default", "price_card", "price_discount",
                "price1_qr", "price2_qr", "price3_qr", "price4_qr",
                "action_price_qr", "wholesale_level_1_price",
                "wholesale_level_2_price", "discount_amount"}
COORD = {"x_min", "y_min", "x_max", "y_max", "frame_timestamp"}
SKIP = {"filename"} | COORD  # not scored for correctness (geometry/meta)


def num(x):
    if x is None:
        return None
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(x))
    return float(m.group().replace(",", ".")) if m else None


def norm(field, v):
    s = "" if v is None else str(v).strip()
    sl = s.casefold()
    if sl in ("нет", "no"):
        return "нет"
    if s == "":
        return ""
    if field in PRICE_FIELDS:
        n = num(s)
        return f"{n:.2f}" if n is not None else sl
    if field == "print_datetime":
        d = re.findall(r"\d+", s)
        return ".".join(d[:5]) if d else sl
    return re.sub(r"\s+", " ", sl)


def field_eq(field, a, b):
    na, nb = norm(field, a), norm(field, b)
    if na == nb:
        return True
    if field in PRICE_FIELDS:
        x, y = num(a), num(b)
        return x is not None and y is not None and abs(x - y) <= 0.01
    if na and nb and na not in ("нет", "") and nb not in ("нет", ""):
        # lenient string: token-set near-equality
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(na, nb) >= 90
    return False


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def key_bc(r):
    for k in ("barcode", "qr_code_barcode"):
        v = (r.get(k) or "").strip()
        if v and v != "нет" and v.isdigit() and len(v) >= 8:
            return v
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--ts-tol", type=float, default=2000)
    ap.add_argument("--ctr-tol", type=float, default=200)
    args = ap.parse_args(argv)

    pred = load(args.pred)
    gt = []
    for c in sorted(glob.glob(os.path.join(args.data, "*", "*.csv"))):
        if os.sep + "Unlabeled" + os.sep in c:
            continue
        zone = os.path.basename(os.path.dirname(c))
        for r in load(c):
            r["_zone"] = zone
            gt.append(r)

    # index predictions
    p_by_bc = defaultdict(list)
    for r in pred:
        bc = key_bc(r)
        if bc:
            p_by_bc[bc].append(r)
    fields = [k for k in (gt[0].keys() if gt else []) if k not in SKIP
              and not k.startswith("_")]

    from rapidfuzz import fuzz

    def field_sim(field, a, b):
        """Lenient partial credit in [0,1] (char/numeric similarity)."""
        na, nb = norm(field, a), norm(field, b)
        if na == nb:
            return 1.0
        if field in PRICE_FIELDS:
            x, y = num(a), num(b)
            if x is not None and y is not None:
                return max(0.0, 1.0 - abs(x - y) / max(1.0, abs(y)))
            return 0.0
        if not na or not nb or na in ("нет",) or nb in ("нет",):
            return 0.0
        return fuzz.ratio(na, nb) / 100.0

    matched = 0
    tag_scores = []          # strict per-tag field-correct fraction
    tag_sims = []            # lenient per-tag mean field similarity
    ferr = defaultdict(lambda: [0, 0])  # field -> [correct, total]
    used = set()
    for g in gt:
        bc = key_bc(g)
        cand = None
        if bc and p_by_bc.get(bc):
            for r in p_by_bc[bc]:
                if id(r) not in used:
                    cand = r
                    break
        if cand is None:
            gt_ts, gx = num(g.get("frame_timestamp")), num(g.get("x_min"))
            gy = num(g.get("y_min"))
            best = None
            for r in pred:
                if id(r) in used:
                    continue
                rt = num(r.get("frame_timestamp"))
                rx, ry = num(r.get("x_min")), num(r.get("y_min"))
                if None in (gt_ts, rt, gx, rx, gy, ry):
                    continue
                if abs(gt_ts - rt) <= args.ts_tol and \
                   ((gx - rx) ** 2 + (gy - ry) ** 2) ** .5 <= args.ctr_tol:
                    best = r
                    break
            cand = best
        if cand is None:
            tag_scores.append(0.0)
            tag_sims.append(0.0)
            for f in fields:
                ferr[f][1] += 1
            continue
        used.add(id(cand))
        matched += 1
        ok = 0.0
        sim = 0.0
        for f in fields:
            good = field_eq(f, g.get(f), cand.get(f))
            ok += good
            sim += field_sim(f, g.get(f), cand.get(f))
            ferr[f][0] += good
            ferr[f][1] += 1
        tag_scores.append(ok / len(fields) if fields else 0.0)
        tag_sims.append(sim / len(fields) if fields else 0.0)

    n = len(gt)
    passed = sum(1 for s in tag_scores if s >= THRESH)
    passed_l = sum(1 for s in tag_sims if s >= THRESH)
    print(f"GT tags: {n} | matched: {matched} "
          f"({matched / n * 100:.1f}%)" if n else "no GT")
    print(f"mean per-tag field acc  (strict) : "
          f"{sum(tag_scores) / n * 100:.1f}%" if n else "")
    print(f"mean per-tag similarity (lenient): "
          f"{sum(tag_sims) / n * 100:.1f}%" if n else "")
    print(f"=== METRIC (strict, exact/fuzzy>=90): tags >=80%: "
          f"{passed}/{n} = {passed / n * 100:.1f}% ===" if n else "")
    print(f"=== METRIC (lenient, char/numeric sim): tags >=80%: "
          f"{passed_l}/{n} = {passed_l / n * 100:.1f}% ===" if n else "")
    print("\nper-field accuracy (worst first):")
    for f, (c, t) in sorted(ferr.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1])):
        print(f"  {f:24s} {c}/{t} = {c / max(1, t) * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
