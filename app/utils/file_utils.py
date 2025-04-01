import re
import fitz
from app.config.settings import MIN_CHUNK_LENGTH
from io import BytesIO


def clean_text_with_mapping(text):
    mapping = []
    cleaned_text = []

    index_in_original = 0

    for char in text:
        if char.isalnum() or char.isspace():
            mapping.append(index_in_original)
            cleaned_text.append(char)
        index_in_original += 1

    cleaned_text = "".join(cleaned_text).lower()
    cleaned_text = re.sub(r"\bngày\s+\d{1,2}\s*tháng\s+\d{1,2}\s*năm\s+\d{4}\b", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\d+(\.\d+)?", "", cleaned_text)
    cleaned_text = re.sub(r"[_\-]", " ", cleaned_text)
    cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)
    cleaned_text = re.sub(r"cộng\s*hòa\s*xã\s*hội\s*chủ\s*nghĩa\s*việt\s*nam[\s\W]*", " ", cleaned_text,
                          flags=re.IGNORECASE)
    cleaned_text = re.sub(r"độc\s*lập\s*-?\s*tự\s*do\s*-?\s*hạnh\s*phúc[\s\W]*", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'"[^"]*"', ' ', cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text, mapping


def extract_text_without_headers_footers(pdf_bytes: BytesIO, skip_pages=None):
    if skip_pages is None:
        skip_pages = set()

    doc = fitz.open("pdf", pdf_bytes.getvalue())
    pages_text = []
    stop_page = None

    for page_num in range(len(doc) - 1, -1, -1):
        page = doc[page_num]
        page_text = page.get_text("text").strip()

        if re.search(r"tài liệu tham khảo", page_text, flags=re.IGNORECASE):
            stop_page = page_num + 1
            print(f"Dừng xử lý tại trang {stop_page} do phát hiện 'Tài liệu tham khảo'")
            break

    for page_num, page in enumerate(doc):
        page_index = page_num + 1
        if page_index in skip_pages:
            continue

        page_rect = page.rect
        page_height = page_rect.height

        header_margin = page_height * 0.08
        footer_margin = page_height * 0.08

        main_rect = fitz.Rect(
            page_rect.x0,
            page_rect.y0 + header_margin,
            page_rect.x1,
            page_rect.y1 - footer_margin
        )

        page_text = page.get_text("text", clip=main_rect).strip()

        if stop_page and page_index == stop_page:
            match = re.search(r"tài liệu tham khảo", page_text, flags=re.IGNORECASE)
            if match:
                page_text = page_text[:match.start()].strip()

        if stop_page and page_index > stop_page:
            continue

        pages_text.append({
            "page": page_index,
            "content": page_text
        })

    doc.close()
    return pages_text


def process_chunks(chunks, metadata_list, min_chunk_length=None):
    processed_chunks = []
    processed_metadata = []

    if min_chunk_length is None:
        min_chunk_length = MIN_CHUNK_LENGTH

    for chunk, metadata in zip(chunks, metadata_list):
        if processed_chunks and len(chunk) < min_chunk_length:
            processed_chunks[-1] += " " + chunk
            processed_metadata[-1]["end"] = metadata["end"]
        else:
            processed_chunks.append(chunk)
            processed_metadata.append(metadata)

    return processed_chunks, processed_metadata
