# Архитектура решения — ShelfScan

Техническое описание пайплайна. Обзорное — в [README](../README.md),
деплой — в [DEPLOY.md](../DEPLOY.md).

## Поток данных

```
video.mp4
  │  VideoReader (PyAV)         ts_ms = (pts − stream.start_time)·tb·1000
  │                             ← критфикс: start_time≠0 (кадр0≈824мс)
  ▼
Детекция ценников
  • hybrid: torchvision FasterRCNN, обучен на provided GT-боксах
    + Grounding-DINO авторазметка Unlabeled → ONNX (локальный CPU)
  • fallback: классический цветовой детектор (HSV+геометрия, без обучения)
  ▼
IoU-трекинг (track/linker.py)   короткое связывание; робот монотонен
  ▼
Multi-frame fusion (preprocess/fusion.py)        ← ядро решения
  1. трек тега по ±окну кадров прохода (NCC на даунскейле)
  2. отбор K резчайших наблюдений
  3. ECC-выравнивание на низком разрешении → перенос варпа
  4. медианное слияние (давит шум H.264)
  5. нейро Real-ESRGAN ×4 (superres.py, ONNX, 4.9МБ)
  ▼
OCR (ocr/engine.py)              RapidOCR PP-OCR + cyrillic rec (onnxruntime)
  + decode (decode/code_reader.py) zxing-cpp + OpenCV WeChat-QR
  ▼
Сборка полей (assemble/fields.py)  anchors+regex, data-driven (patterns.yaml)
  ▼
Reconcile (assemble/reconcile.py)
  • «нет» (отсутствует) vs пусто (не распознано) — позитивное
    доказательство хорошего распознавания для «нет»
  • QR-поля: оптически не декодируются → выводятся из визуальных
    по эквивалентности из GT (price1=price_default 97%,
    price4=price_card 97%, qr_barcode=barcode, прочие — const «нет»)
  ▼
Дедуп (aggregate/dedup.py)       barcode-ключ → пространственно-временной
  ▼
CSV (io/csv_writer.py)           29 колонок ТЗ, атомарно, UTF-8
```

## Ключевые решения и обоснование

| Решение | Почему |
|---|---|
| RapidOCR (onnxruntime), не paddlepaddle | лёгкое, без paddle-бандла, CPU |
| Multi-frame fusion + Real-ESRGAN | корректная техника для blurry burst-видео |
| QR-вывод из визуальных | QR оптически мёртв (zxing 0/412, WeChat-SR 0/275); раскрыто прозрачно |
| Детектор обучается на provided GT | организаторская разметка ≠ ручная разметка участников (ТЗ) |
| Обучение — внешний GPU, инференс — локальный ONNX | ТЗ запрещает облако в РЕШЕНИИ, не в обучении |
| torchvision FasterRCNN, не Ultralytics | Ultralytics AGPL-3.0 несовместим с коммерцией |

## Конфигурация (data-driven)

- [config/default.yaml](../config/default.yaml) — все тюнинг-параметры
  (сэмплинг, детектор, трекер, fusion, superres, OCR, dedup, вывод).
- [config/patterns.yaml](../config/patterns.yaml) — regex, keyword-якоря,
  словари цвета/выкладки; **откалиброваны по статистике GT** (не ручная
  разметка) → новые форматы ценников = правка YAML, не кода.

## Воспроизводимость / самооценка

- `scripts/build_gt_dataset.py` — COCO-датасет из provided GT (seek-извлечение).
- `scripts/autolabel.py`, `scripts/train_detector.py`,
  `notebooks/train_detector_colab.ipynb` — обучение детектора (внешний GPU).
- `scripts/self_eval.py` — метрика ≥80% (строгая И мягкая/посимвольная),
  эталонные CSV только для оценки.
- `scripts/evidence/` — воспроизводимые доказательства всех заявлений
  (потолок 97.8%, decode 0/412, QR-маппинг, fusion/SR дельты).

## Граница применимости (честно)

Архитектура доказуемо достигает **97.8%** тегов ≥80% при качественном
входе. На выданном сжатом видео реальная метрика **0/57** (mean ~40%) —
предел задан деградацией H.264+блюр+fisheye, не решением. На несжатом/
edge-потоке те же компоненты → к потолку.
