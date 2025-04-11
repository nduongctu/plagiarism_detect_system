import re
import fitz
import pytesseract
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes
from app.config.settings import MIN_CHUNK_LENGTH


def clean_text_with_mapping(text, title=None):
    mapping = []
    cleaned_text = []

    index_in_original = 0
    for char in text:
        if char.isalnum() or char.isspace() or char == ".":
            mapping.append(index_in_original)
            cleaned_text.append(char)
        index_in_original += 1

    cleaned_text = "".join(cleaned_text)

    # Xoá các chỉ mục dạng [1], [45], [12]
    cleaned_text = re.sub(r"\[\d+\]", " ", cleaned_text)

    # Xoá ngày tháng
    cleaned_text = re.sub(r"\bngày\s+\d{1,2}\s*tháng\s+\d{1,2}\s*năm\s+\d{4}\b", " ", cleaned_text, flags=re.IGNORECASE)

    # Xoá số (gồm cả số thực)
    cleaned_text = re.sub(r"\d+(\.\d+)?", "", cleaned_text)

    # Xoá cụm "kính gửi", "sở y tế", "hội đồng..."
    cleaned_text = re.sub(r"\bkính\s*gửi\b", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bhội\s*đồng\s*xét\s*công\s*nhận\s*sáng\s*kiến\s*cấp\s*thành\s*phố\b", " ", cleaned_text,
                          flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bsở\s*y\s*tế\b", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bhội\s*đồng\s*xét\s*công\s*nhận\s*sở\s*y\s*tế\s*tp\s*\.?\s*hcm\b", " ", cleaned_text,
                          flags=re.IGNORECASE)

    # Xoá dấu - hoặc chuỗi dấu -
    cleaned_text = re.sub(r"(?:\s*-\s*)+", " ", cleaned_text)

    # Xoá dấu _
    cleaned_text = re.sub(r"_", " ", cleaned_text)

    # Xoá đoạn từ "quyết định nghiệm thu" đến "thuyết minh về phạm vi ảnh hưởng"
    cleaned_text = re.sub(
        r"quyết\s*định\s*nghiệm\s*thu.*?thuyết\s*minh\s*về\s*phạm\s*vi\s*ảnh\s*hưởng",
        " ",
        cleaned_text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Xoá các ký tự đặc biệt (trừ dấu chấm)
    cleaned_text = re.sub(r"[^\w\s.]", " ", cleaned_text)

    cleaned_text = re.sub(r"cộng\s*hòa\s*xã\s*hội\s*chủ\s*nghĩa\s*việt\s*nam[\s\W]*", " ", cleaned_text,
                          flags=re.IGNORECASE)
    cleaned_text = re.sub(r"độc\s*lập\s*tự\s*do\s*hạnh\s*phúc[\s\W]*", " ", cleaned_text, flags=re.IGNORECASE)

    # Rút gọn khoảng trắng
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # Cắt text từ vị trí title (nếu có)
    if title:
        title_cleaned = re.sub(r"\s+", " ", title.lower().strip())
        title_cleaned = re.sub(r"[^\w\s.]", "", title_cleaned)

        index = cleaned_text.find(title_cleaned)
        if index != -1:
            cleaned_text = cleaned_text[index:]

    return cleaned_text, mapping


def remove_signature_tail(text: str) -> str:
    patterns = [r"XÁC NHẬN CỦA", r"NGƯỜI YÊU CẦU CÔNG NHẬN"]
    min_index = len(text)
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            min_index = min(min_index, match.start())
    return text[:min_index].strip()


def extract_text_without_headers_footers(pdf_bytes: BytesIO, skip_pages=None):
    if skip_pages is None:
        skip_pages = set()

    pages_text = []
    raw_bytes = pdf_bytes.getvalue()

    try:
        doc = fitz.open("pdf", raw_bytes)
        test_text = ""
        for i in range(min(2, len(doc))):
            test_text += doc[i].get_text("text").strip() + " "
        test_text = re.sub(r'\s+', ' ', test_text).strip()

        if test_text and len(test_text) >= 15:
            stop_page = None
            for page_num in range(len(doc) - 1, -1, -1):
                page_text = doc[page_num].get_text("text").strip()
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

                if page_num == 0:
                    header_margin = page_height * 0.20
                else:
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

                page_text = remove_signature_tail(page_text)

                pages_text.append({
                    "page": page_index,
                    "content": page_text
                })
            doc.close()
            return pages_text
        else:
            doc.close()
            raise Exception("Không phát hiện đủ văn bản, chuyển qua OCR")
    except Exception as e:
        print(f"Fall back OCR do : {e}")
        images = convert_from_bytes(raw_bytes, dpi=300)
        stop_page = None
        for i in range(len(images) - 1, max(-1, len(images) - 3), -1):
            img_rgb = images[i].convert("RGB")
            text = pytesseract.image_to_string(img_rgb, lang="vie").strip()
            if re.search(r"tài liệu tham khảo", text, flags=re.IGNORECASE):
                stop_page = i
                print(f"Dừng xử lý tại trang {stop_page + 1} do phát hiện 'Tài liệu tham khảo'")
                break

        for idx, img in enumerate(images):
            page_index = idx + 1
            if page_index in skip_pages:
                continue
            if stop_page is not None and idx > stop_page:
                continue

            img_rgb = img.convert("RGB")

            if idx == 0:  # Trang đầu
                width, height = img_rgb.size
                img_rgb = img_rgb.crop((0, int(height * 0.20), width, height))

            text = pytesseract.image_to_string(img_rgb, lang="vie").strip()

            if stop_page is not None and idx == stop_page:
                match = re.search(r"tài liệu tham khảo", text, flags=re.IGNORECASE)
                if match:
                    text = text[:match.start()].strip()

            text = remove_signature_tail(text)

            pages_text.append({
                "page": page_index,
                "content": text
            })
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
