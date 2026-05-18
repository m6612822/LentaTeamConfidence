# Деплой демо — Hugging Face Spaces (бесплатный CPU, публичный URL)

UI ([app.py](app.py)) работает локально и на HF Spaces без изменений.

## Локально
```bash
pip install -r requirements.txt
bash scripts/download_weights.sh           # WeChat-QR + Cyrillic OCR
PYTHONPATH=src python app.py               # http://localhost:7860
```

## Hugging Face Spaces (публичная ссылка, без авторизации)

1. Создать Space: **SDK = Gradio**, hardware = CPU basic (free).
2. Запушить репозиторий в Space (или подключить GitHub).
3. В корень Space положить `README.md` с фронтматтером (ниже) — HF Spaces
   читает его для конфигурации Space; основной проектный README можно
   держать как `README_PROJECT.md`.
4. Веса тянутся при старте: добавить в начало `app.py` запуск
   `scripts/download_weights.sh` ИЛИ закоммитить `models/` через Git LFS.
5. Детектор: без `models/detector.onnx` UI использует классический
   цветовой детектор (демо работает end-to-end). Для лучшего качества —
   обучить `detector.onnx` (см. `notebooks/train_detector_colab.ipynb`) и
   положить в `models/` (Git LFS).

### Фронтматтер для README.md Space
```yaml
---
title: Lenta — распознавание ценников
emoji: 🏷️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---
```

### Лимиты free CPU и митигации
- 2 vCPU / 16 ГБ RAM, sleep после 48ч простоя (прогреть перед демо).
- Видео — короткий клип; multi-frame fusion дорог на CPU →
  для демо `config: fusion.enabled=false` или малый `window_ms`,
  либо платный CPU-upgrade Space (~$0.03/ч) на время защиты.
- Загрузка видео ≤100–200 МБ (ограничение прокси).

## Замечание о метрике на демо
На выданном сжатом видео метрика ≈0 (доказано, см. README) — это предел
данных. Демо показывает корректную работу пайплайна (детекция→fusion→
OCR→CSV) и формат вывода; на качественном входе метрика → потолок 97.8%.
