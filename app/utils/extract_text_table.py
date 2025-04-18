from PIL import Image
import torch
import easyocr
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import numpy as np
from app.utils.table_utils import *

ocr_reader = easyocr.Reader(['vi'], gpu=True)

str_proc = AutoImageProcessor.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all",
    use_fast=True
)

str_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
)

str_model.eval()


def extract_and_process_table_from_image(image: Image, threshold: float = 0.7) -> str:
    table = auto_rotate_image(image)

    w, h = table.size

    inputs = str_proc(
        images=table,
        return_tensors="pt",
        do_resize=False
    )

    with torch.no_grad():
        outputs = str_model(**inputs)

    results = str_proc.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[(h, w)]
    )[0]

    boxes = [[int(x) for x in box.tolist()] for box in results["boxes"]]
    labels = [str_model.config.id2label[i] for i in results["labels"].tolist()]
    elements = [{'box': box, 'label': label} for box, label in zip(boxes, labels)]

    rows = []
    columns = []
    table_structure = []

    for element in elements:
        if element['label'] == 'table row':
            rows.append(element)
        elif element['label'] == 'table column':
            columns.append(element)
        elif element['label'] == 'table':
            table_structure.append(element)

    rows.sort(key=lambda x: (x['box'][1], x['box'][0]))
    columns.sort(key=lambda x: (x['box'][0], x['box'][1]))

    ocr_result = ocr_reader.readtext(np.array(table), detail=1)

    cells = {}
    for row_idx, row_elem in enumerate(rows):
        for col_idx, col_elem in enumerate(columns):
            row_box = row_elem['box']
            col_box = col_elem['box']

            x1 = max(row_box[0], col_box[0])
            y1 = max(row_box[1], col_box[1])
            x2 = min(row_box[2], col_box[2])
            y2 = min(row_box[3], col_box[3])

            if x1 < x2 and y1 < y2:
                cell_key = (row_idx, col_idx)
                cells[cell_key] = {
                    'row': row_idx,
                    'col': col_idx,
                    'text': [],
                    'box': [x1, y1, x2, y2]
                }

    if not cells:
        return ""

    for detection in ocr_result:
        text = detection[1]
        coords = detection[0]

        x_coords = [pt[0] for pt in coords]
        y_coords = [pt[1] for pt in coords]
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        for (row_idx, col_idx), cell in cells.items():
            x1, y1, x2, y2 = cell['box']
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                cell['text'].append((text, center_y, center_x))
                break

    result_table = []
    for (row_idx, col_idx), cell in cells.items():
        sorted_texts = sorted(cell['text'], key=lambda t: (t[1], t[2]))
        full_text = ' '.join([t[0].strip() for t in sorted_texts])
        result_table.append({
            'row': row_idx,
            'col': col_idx,
            'text': full_text
        })

    rows_data = {}
    for cell in result_table:
        row_idx = cell['row']
        col_idx = cell['col']
        text = cell['text'].strip()
        if row_idx not in rows_data:
            rows_data[row_idx] = {}
        rows_data[row_idx][col_idx] = text

    if not rows_data or not any(rows_data.values()):
        return ""

    try:
        max_col = max(col_idx for row in rows_data.values() for col_idx in row.keys())
    except ValueError:
        return ""

    header_row = rows_data.get(0, {})

    table_text = []
    for row_idx in sorted(rows_data.keys()):
        if row_idx == 0:
            continue  # Bỏ qua header
        row_data = rows_data[row_idx]
        line_parts = []
        for col_idx in range(max_col + 1):
            header_text = header_row.get(col_idx, "").lower()
            cell_text = row_data.get(col_idx, "").strip()
            if header_text:
                line_parts.append(f"{header_text} {cell_text}")
            else:
                line_parts.append(cell_text)
        line = ' '.join(line_parts)
        table_text.append(line)

    return ' '.join(table_text)
