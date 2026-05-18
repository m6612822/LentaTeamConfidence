"""Gradio UI: upload robot video -> recognized price-tag CSV.

Deployable to Hugging Face Spaces (free CPU). Local:
    PYTHONPATH=src .venv/bin/python app.py

Detector mode needs models/detector.onnx (trained once on a free GPU, see
notebooks/train_detector_colab.ipynb). Without it, falls back to the
classical colour detector so the demo still runs end-to-end.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import pandas as pd

from lenta.config import load_config
from lenta.io.csv_writer import write_csv
from lenta.pipeline import Pipeline
from lenta.schema import CSV_COLUMNS

CFG = load_config()
PIPE = Pipeline(CFG, models_dir="models")


def _detector():
    onnx = CFG["detector"]["weights"]
    if os.path.exists(onnx):
        from lenta.detect.detector import ONNXDetector
        return ONNXDetector(CFG), "trained ONNX detector"
    from lenta.detect.color_detector import ColorTagDetector
    return ColorTagDetector(CFG), "classical colour detector (no trained weights)"


def process(video_path: str, progress=gr.Progress()):
    if not video_path:
        return None, None, "Загрузите видео."
    progress(0.1, desc="Инициализация детектора")
    detector, note = _detector()
    progress(0.3, desc=f"Обработка видео ({note})")
    rows = PIPE.run_detect(video_path, detector,
                           out_filename=os.path.basename(video_path))
    out = os.path.join(tempfile.mkdtemp(), "result.csv")
    n = write_csv(rows, out)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS) if rows \
        else pd.DataFrame(columns=CSV_COLUMNS)
    progress(1.0, desc="Готово")
    return out, df, f"Распознано ценников: {n}. Детектор: {note}."


with gr.Blocks(title="Lenta — распознавание ценников") as demo:
    gr.Markdown("# Распознавание ценников с видео робота\n"
                "Загрузите видео стеллажа — получите CSV с полями ценников.")
    with gr.Row():
        inp = gr.Video(label="Видео с робота", sources=["upload"])
        with gr.Column():
            btn = gr.Button("Распознать", variant="primary")
            status = gr.Textbox(label="Статус", interactive=False)
            csv_out = gr.File(label="Скачать CSV")
    table = gr.Dataframe(label="Результат (превью)", wrap=True)
    btn.click(process, inputs=inp, outputs=[csv_out, table, status])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0",
                        server_port=int(os.environ.get("PORT", 7860)))
