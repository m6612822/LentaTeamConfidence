#!/usr/bin/env bash
# Fetch model weights into models/ (NOT committed). Idempotent.
#   - OpenCV WeChatQRCode (detector + super-resolution) for blurry QR
#   - Cyrillic PP-OCR recognition model + dict (RapidOCR) for product_name
#   - detector ONNX is produced by training (see scripts/autolabel_distill.py)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/wechat_qr models/ocr_cyrillic

dl() { # url dest
  [ -f "$2" ] && { echo "  exists $2"; return; }
  echo "  get $2"
  curl -fL --retry 3 -o "$2" "$1" || wget -q -O "$2" "$1"
}

echo "[1/2] WeChatQRCode (Apache-2.0, opencv_3rdparty)"
B="https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode"
dl "$B/detect.prototxt"   models/wechat_qr/detect.prototxt
dl "$B/detect.caffemodel" models/wechat_qr/detect.caffemodel
dl "$B/sr.prototxt"       models/wechat_qr/sr.prototxt
dl "$B/sr.caffemodel"     models/wechat_qr/sr.caffemodel

echo "[2/2] Cyrillic PP-OCRv3 rec ONNX + dict (HF, Apache-2.0)"
# Verified-reachable source (PP-OCRv3 cyrillic, ONNX). Override with
# CYRILLIC_REC_URL / CYRILLIC_KEYS_URL if mirrored elsewhere.
HF="https://huggingface.co/cycloneboy/cyrillic_PP-OCRv3_rec_infer/resolve/main"
REC="${CYRILLIC_REC_URL:-$HF/model.onnx}"
KEYS="${CYRILLIC_KEYS_URL:-$HF/cyrillic_dict.txt}"
dl "$REC"  models/ocr_cyrillic/rec_cyrillic.onnx  || \
  echo "  WARN: Cyrillic rec not fetched — product_name weak without it."
dl "$KEYS" models/ocr_cyrillic/cyrillic_dict.txt || true

echo "[3/3] Real-ESRGAN general-x4-v3 ONNX (BSD-3) — neural upscale"
mkdir -p models/superres
SR="${REALESRGAN_URL:-https://huggingface.co/Samo629/real-esrgan-onnx/resolve/main/realesr-general-x4v3.onnx}"
dl "$SR" models/superres/realesr-general-x4v3.onnx || \
  echo "  WARN: Real-ESRGAN not fetched — fusion falls back to cubic upscale."

echo "done. (detector.onnx comes from training — see README 'Авторазметка')"
