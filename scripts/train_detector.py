"""Train the lightweight price-tag detector on the HYBRID COCO dataset
(GT boxes + Grounding-DINO auto-labels) and export ONNX.

torchvision FasterRCNN-ResNet50-FPN (license: BSD-3) — robust on small
data, exports to ONNX as (boxes, scores, labels) which matches
lenta.detect.detector.ONNXDetector convention B. Ultralytics YOLO is
deliberately NOT used (AGPL-3.0).

Run on a GPU machine (Colab/Kaggle); produces models/detector.onnx that
runs locally on CPU via onnxruntime. NO cloud at inference (ТЗ-compliant).

  python scripts/build_gt_dataset.py --data data --out dataset --val-zones 49_5
  python scripts/autolabel.py --data data --out dataset      # optional, GPU
  python scripts/train_detector.py --dataset dataset --epochs 30 \
         --out models/detector.onnx
"""
from __future__ import annotations

import argparse
import json
import os

import torch


class CocoDet(torch.utils.data.Dataset):
    def __init__(self, root, ann, train):
        import cv2
        self.cv2 = cv2
        self.root = root
        d = json.load(open(ann, encoding="utf-8"))
        self.imgs = {i["id"]: i for i in d["images"]}
        self.by_img = {}
        for a in d["annotations"]:
            self.by_img.setdefault(a["image_id"], []).append(a["bbox"])
        self.ids = [i for i in self.imgs if self.by_img.get(i)]
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        iid = self.ids[k]
        im = self.imgs[iid]
        img = self.cv2.imread(os.path.join(self.root, im["file_name"]))
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        boxes = [[x, y, x + w, y + h] for x, y, w, h in self.by_img[iid]]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if self.train and torch.rand(1).item() < 0.5:        # hflip
            img = img[:, ::-1, :].copy()
            W = img.shape[1]
            boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) \
                + torch.tensor([W, 0, W, 0])
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {"boxes": boxes,
                  "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return t, target


def collate(b):
    return tuple(zip(*b))


class _ExportWrap(torch.nn.Module):
    """Single-image forward -> (boxes, scores, labels) for ONNX."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)[0]
        return o["boxes"], o["scores"], o["labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--out", default="models/detector.onnx")
    a = ap.parse_args()

    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir = os.path.join(a.dataset, "images")
    tr = CocoDet(img_dir,
                 os.path.join(a.dataset, "annotations/instances_train.json"),
                 True)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True,
                                     collate_fn=collate, num_workers=2)
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    inf = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(inf, 2)  # bg + tag
    model.to(dev)

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for imgs, tgts in dl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        print(f"epoch {ep+1}/{a.epochs} loss={tot/max(1,len(dl)):.3f}")

    model.eval()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dummy = torch.rand(1, 3, 960, 960).to(dev)
    torch.onnx.export(
        _ExportWrap(model).eval(), dummy, a.out,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={"images": {0: "b", 2: "h", 3: "w"},
                      "boxes": {0: "n"}, "scores": {0: "n"},
                      "labels": {0: "n"}},
        opset_version=17,
        dynamo=False)
    print(f"exported -> {a.out}")


if __name__ == "__main__":
    main()
