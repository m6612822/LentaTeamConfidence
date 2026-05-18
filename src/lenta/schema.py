"""Single source of truth for the output CSV schema (ТЗ «Образ результата»).

Column order is fixed by the task spec and must not change.
"""
from __future__ import annotations

# Visual fields read from the price tag (OCR + detection geometry).
VISUAL_FIELDS = [
    "filename",          # video file name
    "product_name",      # наименование товара
    "price_default",     # цена без карты
    "price_card",        # цена по карте
    "price_discount",    # цена по акции
    "barcode",           # штрихкод (visual)
    "discount_amount",   # размер скидки
    "id_sku",            # артикул
    "print_datetime",    # дата и время печати
    "code",              # код зоны выкладки
    "additional_info",   # доп. информация с ценника
    "color",             # цвет ценника
    "special_symbols",   # тип выкладки
    "frame_timestamp",   # ms from video start
    "x_min",             # bbox top-left x (px)
    "y_min",             # bbox top-left y (px)
    "x_max",             # bbox bottom-right x (px)
    "y_max",             # bbox bottom-right y (px)
]

# Structured fields decoded from the tag's QR code.
QR_FIELDS = [
    "qr_code_barcode",          # barcode | b
    "price1_qr",                # price1 | p1
    "price2_qr",                # price2 | p2
    "price3_qr",                # price3 | p3
    "price4_qr",                # price4 | p4
    "wholesale_level_1_count",  # wholesaleLevel1Count | wL1C
    "wholesale_level_1_price",  # wholesaleLevel1Price | wL1P
    "wholesale_level_2_count",  # wholesaleLevel2Count | wL2C
    "wholesale_level_2_price",  # wholesaleLevel2Price | wL2P
    "action_price_qr",          # actionPrice | aP
    "action_code_qr",           # actionCode | aC
]

CSV_COLUMNS = VISUAL_FIELDS + QR_FIELDS  # 29 columns, exact spec order

# Per ТЗ: parameter physically ABSENT on the tag -> literal "нет".
#         parameter present but NOT recognized -> empty string.
ABSENT = "нет"
UNREAD = ""

assert len(CSV_COLUMNS) == 29, "CSV schema must be exactly 29 columns"
assert len(set(CSV_COLUMNS)) == 29, "CSV columns must be unique"
