"""Quick recall/precision of a detector vs GT COCO (no training needed)."""
import json, sys, glob, os
sys.path.insert(0, "src")
import cv2
from lenta.config import load_config
from lenta.detect.color_detector import ColorTagDetector
from lenta.types import BBox

cfg = load_config()
det = ColorTagDetector(cfg)

for split in ("train", "val"):
    d = json.load(open(f"dataset/annotations/instances_{split}.json"))
    by_img = {im["id"]: im for im in d["images"]}
    gt = {}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        gt.setdefault(a["image_id"], []).append(BBox(x, y, x + w, y + h))
    TP = FP = FN = 0
    for iid, im in by_img.items():
        img = cv2.imread(os.path.join("dataset/images", im["file_name"]))
        if img is None:
            continue
        preds = [dd.bbox for dd in det.detect(img, 0, 0.0)]
        g = gt.get(iid, [])
        matched = set()
        for p in preds:
            hit = False
            for i, gb in enumerate(g):
                if i in matched:
                    continue
                if p.iou(gb) >= 0.3:        # loose IoU: best-frame uses crop
                    matched.add(i)
                    hit = True
                    break
            TP += hit
            FP += not hit
        FN += len(g) - len(matched)
    rec = TP / max(1, TP + FN)
    prec = TP / max(1, TP + FP)
    print(f"{split}: imgs={len(by_img)} GT={TP+FN} | "
          f"recall={rec*100:.1f}% precision={prec*100:.1f}% "
          f"(TP={TP} FP={FP} FN={FN})")
