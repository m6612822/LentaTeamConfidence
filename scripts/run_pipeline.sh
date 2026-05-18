#!/usr/bin/env bash
# Video(s) -> CSV. One-command reproducible entrypoint.
#   bash scripts/run_pipeline.sh <input> <out.csv> [mode] [config]
# input : a video file OR a directory of videos
# mode  : onnx (default) | zeroshot | gt   (gt = eval on labeled data/)
set -euo pipefail
cd "$(dirname "$0")/.."

INPUT="${1:?usage: run_pipeline.sh <input> <out.csv> [mode] [config]}"
OUT="${2:-out.csv}"
MODE="${3:-onnx}"
CONFIG="${4:-}"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

ARGS=(--input "$INPUT" --out "$OUT" --mode "$MODE" --models models)
[ -n "$CONFIG" ] && ARGS+=(--config "$CONFIG")

PYTHONPATH=src "$PY" -m lenta.cli "${ARGS[@]}"
