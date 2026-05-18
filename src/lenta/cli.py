"""CLI: video(s) -> CSV.

  python -m lenta.cli --input data --mode gt   --out out.csv
  python -m lenta.cli --input clip.mp4 --mode onnx --out out.csv

gt mode (eval/tuning): expects data/<zone>/{<zone>.csv,<zone>.mp4} and
uses the reference CSV boxes as oracle detections.
onnx/zeroshot: real detection on arbitrary video(s).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from lenta.config import load_config
from lenta.io.csv_writer import write_csv
from lenta.pipeline import Pipeline


def _videos(path: str):
    if os.path.isfile(path):
        return [path]
    vids = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        vids += glob.glob(os.path.join(path, "**", ext), recursive=True)
    return sorted(v for v in vids if os.sep + "Unlabeled" + os.sep not in v)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lenta")
    ap.add_argument("--input", required=True, help="video file or directory")
    ap.add_argument("--out", default="out.csv")
    ap.add_argument("--mode", choices=["gt", "onnx", "zeroshot"],
                    default="onnx")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default="models")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    pipe = Pipeline(cfg, args.models)
    vids = _videos(args.input)
    if not vids:
        print(f"no videos under {args.input}", file=sys.stderr)
        return 2

    detector = None
    if args.mode == "onnx":
        from lenta.detect.detector import ONNXDetector
        detector = ONNXDetector(cfg)
    elif args.mode == "zeroshot":
        from lenta.detect.zeroshot import GroundingDINODetector
        detector = GroundingDINODetector(cfg, args.models)

    all_rows = []
    for v in vids:
        if args.mode == "gt":
            ref = os.path.join(os.path.dirname(v),
                               os.path.basename(os.path.dirname(v)) + ".csv")
            if not os.path.exists(ref):
                cands = glob.glob(os.path.join(os.path.dirname(v), "*.csv"))
                ref = cands[0] if cands else None
            if not ref:
                print(f"[skip] no reference CSV for {v}", file=sys.stderr)
                continue
            rows = pipe.run_gt(v, ref, out_filename=os.path.basename(v))
        else:
            rows = pipe.run_detect(v, detector,
                                   out_filename=os.path.basename(v))
        print(f"{os.path.basename(v)}: {len(rows)} rows")
        all_rows += rows

    n = write_csv(all_rows, args.out)
    print(f"wrote {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
