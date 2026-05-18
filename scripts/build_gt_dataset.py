"""Build a COCO detection dataset from the organizer-provided GT CSVs.

This is the supervised half of the HYBRID detection strategy (GT boxes +
Grounding-DINO auto-labels). The CSVs are organizer-provided ground truth,
not participant manual annotation -> ТЗ-compliant.

For each labeled folder data/<zone>/{<zone>.csv,<zone>.mp4}:
  - group GT rows by frame_timestamp (ms from start)
  - decode the video ONCE; for each needed timestamp keep the nearest frame
  - emit the frame as an image + all its tag bboxes (native 3840x2160 px)

Single category: price_tag. Train/val split is BY VIDEO (no frame leakage);
the hidden control video is the real test set.

Usage:
  python scripts/build_gt_dataset.py --data data --out dataset \
         --val-zones 49_5
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lenta.io.video_reader import VideoReader  # noqa: E402


def fnum(x):
    try:
        return float(str(x).replace(",", "."))
    except (ValueError, TypeError):
        return None


def collect_targets(rows):
    """ts(ms) -> list of (x0,y0,x1,y1) valid boxes."""
    targets = {}
    for r in rows:
        ts = fnum(r.get("frame_timestamp"))
        x0, y0 = fnum(r.get("x_min")), fnum(r.get("y_min"))
        x1, y1 = fnum(r.get("x_max")), fnum(r.get("y_max"))
        if ts is None or None in (x0, y0, x1, y1):
            continue
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        targets.setdefault(round(ts), []).append((x0, y0, x1, y1))
    return targets


def nearest_frames(video: str, ts_list):
    """Seek to each target ts (O(targets), not full 4K decode)."""
    vr = VideoReader(video)
    out = {}
    for ts in ts_list:
        fr = vr.frame_at_ms(float(ts))
        if fr is not None:
            out[ts] = fr.image
    return vr, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-zones", nargs="*", default=["49_5"],
                    help="zone folder names held out for validation")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    cat = [{"id": 1, "name": "price_tag"}]
    coco = {s: {"images": [], "annotations": [], "categories": cat}
            for s in ("train", "val")}
    img_id = ann_id = 0

    csvs = sorted(glob.glob(os.path.join(args.data, "*", "*.csv")))
    csvs = [c for c in csvs if os.sep + "Unlabeled" + os.sep not in c]
    for csv_path in csvs:
        zone = os.path.basename(os.path.dirname(csv_path))
        mp4 = os.path.join(os.path.dirname(csv_path), zone + ".mp4")
        if not os.path.exists(mp4):
            cands = glob.glob(os.path.join(os.path.dirname(csv_path), "*.mp4"))
            mp4 = cands[0] if cands else None
        if not mp4:
            print(f"[skip] no video for {csv_path}")
            continue
        with open(csv_path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        targets = collect_targets(rows)
        split = "val" if zone in args.val_zones else "train"
        print(f"{zone}: {len(rows)} rows, {len(targets)} frames -> {split}")
        _, frames = nearest_frames(mp4, list(targets.keys()))
        for ts, boxes in targets.items():
            img = frames.get(ts)
            if img is None:
                continue
            h, w = img.shape[:2]
            img_id += 1
            fn = f"{zone}__{ts}.jpg"
            cv2.imwrite(os.path.join(args.out, "images", fn), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            coco[split]["images"].append(
                {"id": img_id, "file_name": fn, "width": w, "height": h})
            for (x0, y0, x1, y1) in boxes:
                x0, y0 = max(0, min(x0, w)), max(0, min(y0, h))
                x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
                bw, bh = x1 - x0, y1 - y0
                if bw < 2 or bh < 2:
                    continue
                ann_id += 1
                coco[split]["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": [round(x0, 1), round(y0, 1),
                             round(bw, 1), round(bh, 1)],
                    "area": round(bw * bh, 1), "iscrowd": 0})

    for s in ("train", "val"):
        p = os.path.join(args.out, "annotations", f"instances_{s}.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(coco[s], fh, ensure_ascii=False)
        print(f"{s}: {len(coco[s]['images'])} imgs, "
              f"{len(coco[s]['annotations'])} boxes -> {p}")


if __name__ == "__main__":
    main()
