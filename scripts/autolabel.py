"""Auto-label Unlabeled videos with Grounding DINO -> COCO pseudo-labels,
the augmentation half of the HYBRID detector dataset.

Automatic method (no manual annotation). Runs on a GPU machine
(Colab/Kaggle) alongside training; CPU works but is slow.

  python scripts/autolabel.py --data data --out dataset \
         --stride 15 --score 0.35
Appends images+annotations into dataset/ (merging with the GT split from
scripts/build_gt_dataset.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lenta.io.video_reader import VideoReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--prompt",
                    default="price tag . price label . shelf label . ценник .")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForZeroShotObjectDetection,
                              AutoProcessor)
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model).to(dev).eval()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    ann_path = os.path.join(args.out, "annotations", "instances_train.json")
    coco = (json.load(open(ann_path, encoding="utf-8"))
            if os.path.exists(ann_path)
            else {"images": [], "annotations": [],
                  "categories": [{"id": 1, "name": "price_tag"}]})
    img_id = max([i["id"] for i in coco["images"]], default=0)
    ann_id = max([a["id"] for a in coco["annotations"]], default=0)

    vids = sorted(glob.glob(os.path.join(args.data, "Unlabeled", "*.mp4")))
    for v in vids:
        vr = VideoReader(v)
        tag = os.path.splitext(os.path.basename(v))[0]
        for fr in vr.frames():
            if fr.idx % args.stride:
                continue
            pil = Image.fromarray(fr.image[:, :, ::-1])
            inp = proc(images=pil, text=args.prompt,
                       return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inp)
            res = proc.post_process_grounded_object_detection(
                out, inp["input_ids"], box_threshold=args.score,
                text_threshold=args.text_thr,
                target_sizes=[pil.size[::-1]])[0]
            boxes = res["boxes"].tolist()
            if not boxes:
                continue
            img_id += 1
            fn = f"auto_{tag}__{fr.idx}.jpg"
            cv2.imwrite(os.path.join(img_dir, fn), fr.image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = fr.image.shape[:2]
            coco["images"].append({"id": img_id, "file_name": fn,
                                   "width": w, "height": h})
            for (x0, y0, x1, y1) in boxes:
                bw, bh = x1 - x0, y1 - y0
                if bw < 4 or bh < 4:
                    continue
                ann_id += 1
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": [round(x0, 1), round(y0, 1),
                             round(bw, 1), round(bh, 1)],
                    "area": round(bw * bh, 1), "iscrowd": 0})
        print(f"{os.path.basename(v)}: total imgs now {len(coco['images'])}")

    with open(ann_path, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False)
    print(f"merged -> {ann_path}: {len(coco['images'])} imgs, "
          f"{len(coco['annotations'])} boxes")


if __name__ == "__main__":
    main()
