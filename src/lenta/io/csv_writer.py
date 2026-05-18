"""Schema-locked, atomic CSV writer.

Format per ТЗ:
- field delimiter: comma
- decimal separator: dot
- encoding: UTF-8
- string fields containing commas/quotes wrapped in double quotes
"""
from __future__ import annotations

import csv
import os
import tempfile
from typing import Dict, Iterable

from lenta.schema import CSV_COLUMNS


def _fmt(value) -> str:
    """Render a cell. Floats use dot decimal; None -> empty string."""
    if value is None:
        return ""
    if isinstance(value, float):
        # dot decimal, trim trailing zeros but keep integers clean
        s = f"{value:.4f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(value)


def write_csv(rows: Iterable[Dict[str, object]], out_path: str) -> int:
    """Write rows (one dict per unique tag) atomically. Returns row count.

    Unknown keys are ignored; missing keys become empty cells. Column order
    is enforced from `schema.CSV_COLUMNS` regardless of dict insertion order.
    """
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=CSV_COLUMNS,
                extrasaction="ignore",
                quoting=csv.QUOTE_MINIMAL,  # quotes only when needed (, " \n)
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _fmt(row.get(k)) for k in CSV_COLUMNS})
                n += 1
        os.replace(tmp, out_path)  # atomic
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return n
