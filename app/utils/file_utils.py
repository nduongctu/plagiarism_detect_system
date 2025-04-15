import re
import fitz
import pytesseract
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes
from app.config.settings import MIN_CHUNK_LENGTH


def clean_text_with_mapping(text):
    mapping = []
    cleaned_chars = []

    # 1. Nối các dòng lại với nhau trước khi làm sạch
    text = re.sub(r"[\n\r]+", " ", text)  # Thay thế tất cả các dòng mới bằng một khoảng trắng

    # Tạo mảng mapping và chuỗi cleaned_text ban đầu
    index_in_original = 0
    for char in text:
        if char.isalnum() or char.isspace() or char in ".[]":
            mapping.append(index_in_original)
            cleaned_chars.append(char)
        index_in_original += 1
    cleaned_text = "".join(cleaned_chars)

    # 2. Xoá ngày tháng năm (viết bằng số hoặc chữ)
    cleaned_text = re.sub(
        r"\bngày\s+[^\s]+\s+tháng\s+[^\s]+\s+năm\s+\d{4}\b",
        " ",
        cleaned_text,
        flags=re.IGNORECASE
    )

    # 3. Xoá các cụm "Mục tiêu nghiên cứu", "Thiết kế nghiên cứu", "Dân số chọn mẫu"
    cleaned_text = re.sub(
        r"\b(mục\s*tiêu\s*nghiên\s*cứu|thiết\s*kế\s*nghiên\s*cứu|dân\s*số\s*chọn\s*mẫu)\b",
        " ",
        cleaned_text,
        flags=re.IGNORECASE
    )

    # 4. Xoá các cụm hành chính
    cleaned_text = re.sub(r"\bkính\s*gửi\b", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bhội\s*đồng\s*xét\s*công\s*nhận\s*sáng\s*kiến\s*cấp\s*thành\s*phố\b", " ", cleaned_text,
                          flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bsở\s*y\s*tế\b", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bhội\s*đồng\s*xét\s*công\s*nhận\s*sở\s*y\s*tế\s*tp\.?\s*hcm\b", " ", cleaned_text,
                          flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bhội\s*đồng\s*xét\s*công\s*nhận\s*sáng\s*kiến\s*tp\.?\s*hcm\b", " ", cleaned_text,
                          flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bBỆNH\s+VIỆN\b", " ", cleaned_text)

    # 5. Xoá dấu gạch nối và gạch dưới
    cleaned_text = re.sub(r"(?:\s*-\s*)+", " ", cleaned_text)
    cleaned_text = re.sub(r"_", " ", cleaned_text)

    # 6. Xoá các chỉ mục dạng [số] (không có khoảng trắng)
    cleaned_text = re.sub(r"\[\d+\]", " ", cleaned_text)

    # 7. Xoá ký tự đặc biệt (giữ lại chữ, số và khoảng trắng)
    cleaned_text = re.sub(r"[^\w\s]", " ", cleaned_text)

    # 8. Xoá các cụm khẩu hiệu
    cleaned_text = re.sub(r"\bCỘNG\s*HOÀ\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM\b", " ", cleaned_text,
                          flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bTHÀNH\s*PHÓ\s*HỎ\s*CHÍ\s*MINH\b", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bThành\s*phố\s*Hô\s*Chí\s*Minh\b", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bĐộc\s*lập\s*Tự\s*do\s*Hạnh\s*phúc\b", " ", cleaned_text, flags=re.IGNORECASE)

    # 9. Rút gọn khoảng trắng
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # 10. Xoá cụm liên quan đến nghiệm thu và mô tả đề tài
    cleaned_text = re.sub(r"\bđã\s+được\s+nghi[êe]m\s+thu\b", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"tên\s*đề\s*tà[i1]\s*nghiên\s*cứu", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(
        r"mô\s*tả\s*tóm\s*tắt\s*nội\s*dung\s*của\s*đề\s*tà[i1]?\s*nghiên\s*cứu(\s*khoa\s*học)?(\s*chữ)?", " ",
        cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"200\s*500\s*chữ", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # 11. Loại bỏ các cụm tiêu đề của sáng kiến:
    cleaned_text = re.sub(
        r"m[oó]\s*tả\s*t[oó]m\s*tắt\s*n[oó]i\s*dung\s*của\s*s[aá]ng\s*kiến",
        " ",
        cleaned_text,
        flags=re.IGNORECASE
    )
    cleaned_text = re.sub(
        r"b[oỏ]\s*cảnh\s*dẫn\s*t[óo]i\s*s[aá]ng\s*kiến",
        " ",
        cleaned_text,
        flags=re.IGNORECASE
    )
    cleaned_text = re.sub(
        r"\bMẪU\s+SỐ\b",
        " ",
        cleaned_text
    )
    cleaned_text = re.sub(
        r"t[eê]n\s*s[aá]ng\s*ki[eế]n",
        " ",
        cleaned_text,
        flags=re.IGNORECASE
    )
    cleaned_text = re.sub(
        r"n[oộ]i\s*dung\s*s[aá]ng\s*ki[eế]n",
        " ",
        cleaned_text,
        flags=re.IGNORECASE
    )
    cleaned_text = re.sub(
        r"m[oô]\s*t[aả]\s*k[ỹy]\s*thu[aậ]t",
        " ",
        cleaned_text,
        flags=re.IGNORECASE
    )

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text, mapping


def remove_signature_tail(pages_text: list) -> list:
    # Kiểm tra nếu đầu vào là chuỗi (string), chuyển đổi thành định dạng danh sách từ điển
    if isinstance(pages_text, str):
        # Tạo một đối tượng kiểu Dictionary với cấu trúc phù hợp
        pages_text = [{"page": 1, "content": pages_text}]

    # Các pattern xác định phần văn bản không cần thiết (ví dụ: phần xác nhận, tóm tắt)
    patterns = [r"XÁC NHẬN CỦA", r"NGƯỜI YÊU CẦU CÔNG NHẬN"]
    stop_pattern = r"TÓM\s*TẮT"

    # Kết hợp nội dung của tất cả các trang thành một chuỗi để kiểm tra
    combined_text = " ".join(p["content"] for p in pages_text)
    combined_text_norm = normalize_text(combined_text)

    start_idx = None
    for pattern in patterns:
        match = re.search(pattern, combined_text_norm, flags=re.IGNORECASE)
        if match:
            start_idx = match.start()
            break

    if start_idx is None:
        return pages_text  # Không có văn bản cần xóa, trả về kết quả ban đầu

    last_page_content = normalize_text(pages_text[-1]["content"])
    last_page_start = len(normalize_text(" ".join(p["content"] for p in pages_text[:-1])))

    # Xử lý khi mẫu xác nhận được tìm thấy
    if start_idx >= last_page_start:
        # Nếu mẫu tìm thấy trên trang cuối, cắt từ vị trí này trở đi
        end_idx = len(combined_text_norm)
    else:
        # Nếu mẫu tìm thấy ở giữa các trang, tìm đến "TÓM TẮT"
        stop_match = re.search(stop_pattern, combined_text_norm[start_idx:], flags=re.IGNORECASE)
        if stop_match:
            end_idx = start_idx + stop_match.start()
        else:
            end_idx = len(combined_text_norm)

    # Cắt nội dung đã lọc
    cleaned_text = combined_text_norm[:start_idx] + combined_text_norm[end_idx:]

    # Phân trang lại với việc giữ lại số trang và nội dung đã lọc
    result_pages = []
    current_pos = 0
    for p in pages_text:
        # Lấy độ dài của nội dung trang hiện tại đã chuẩn hóa
        length = len(normalize_text(p["content"]))
        segment = cleaned_text[current_pos:current_pos + length].strip()

        # Đảm bảo số trang vẫn được giữ lại trong kết quả
        result_pages.append({
            "page": p["page"],
            "content": segment
        })

        current_pos += length

    return result_pages


def extract_main_text_from_pdf(pdf_bytes: BytesIO, skip_pages=None):
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
                    header_margin = page_height * 0.02

                footer_margin = page_height * 0.02

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

                # Thêm vào danh sách trước khi xử lý xóa chữ ký
                pages_text.append({
                    "page": page_index,
                    "content": page_text
                })
            doc.close()

            # Xử lý xóa chữ ký và phần tài liệu không liên quan
            pages_text = remove_signature_tail(pages_text)
            pages_text = remove_irrelevant_section(pages_text)
            return pages_text
        else:
            doc.close()
            raise Exception("Không phát hiện đủ văn bản, chuyển qua OCR")
    except Exception as e:
        print(f"Fall back OCR do : {e}")
        images = convert_from_bytes(raw_bytes, dpi=300)
        pages_text = []
        stop_index = None
        appendix_text = None
        appendix_page = None

        for idx, img in enumerate(images):
            page_index = idx + 1
            if page_index in skip_pages:
                continue

            img_rgb = img.convert("RGB")
            if idx == 0:
                width, height = img_rgb.size
                img_rgb = img_rgb.crop((0, int(height * 0.20), width, height))

            text = pytesseract.image_to_string(img_rgb, lang="vie").strip()

            # Kiểm tra nếu là trang cuối và chứa "PHỤ LỤC"
            if idx == len(images) - 1 and re.search(r"PHỤ LỤC", text, flags=re.IGNORECASE):
                # Tạo một dictionary cho appendix_text thay vì string
                appendix_text = text
                appendix_page = page_index
                continue  # chưa thêm vào, để sau xử lý

            if stop_index is None:
                match = re.search(r"TÀI LIỆU THAM KHẢO", text)
                if match:
                    text = text[:match.start()].strip()
                    stop_index = idx
                    print(f"Phát hiện 'TÀI LIỆU THAM KHẢO' tại trang {page_index}, dừng sau trang này.")

            if stop_index is not None and idx > stop_index:
                continue  # Bỏ qua các trang sau khi gặp 'TÀI LIỆU THAM KHẢO'

            # Thêm vào danh sách trang
            pages_text.append({
                "page": page_index,
                "content": text
            })

        # Sau cùng, nếu có phụ lục thì thêm vào
        if appendix_text and appendix_page not in [p["page"] for p in pages_text]:
            pages_text.append({
                "page": appendix_page,
                "content": appendix_text
            })

        # Xử lý xóa chữ ký và phần tài liệu không liên quan
        pages_text = remove_signature_tail(pages_text)
        pages_text = remove_irrelevant_section(pages_text)
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


def remove_irrelevant_section(pages_text):
    start_patterns = [
        r"Quyết\s*định\s*nghiệm\s*thu\s*[:：]?\s*số",
        r"Quyết\s*định\s*công\s*nhận\s*Sáng\s*kiến\s*[:：]?\s*số"
    ]
    end_pattern = r"Thuyết\s*minh\s*về\s*phạm\s*vi\s*ảnh\s*hưởng"

    start_idx = None
    end_idx = None
    combined_text = " ".join(p["content"] for p in pages_text)
    combined_text_norm = normalize_text(combined_text)

    for pattern in start_patterns:
        match = re.search(pattern, combined_text_norm, flags=re.IGNORECASE)
        if match:
            start_idx = match.start()
            break

    # Kiểm tra phần kết thúc
    if start_idx is not None:
        end_match = re.search(end_pattern, combined_text_norm[start_idx:], flags=re.IGNORECASE)
        if end_match:
            end_idx = start_idx + end_match.end()

    if start_idx is not None and end_idx is not None:
        cleaned_text = combined_text_norm[:start_idx] + combined_text_norm[end_idx:]
        result_pages = []
        current_pos = 0
        for p in pages_text:
            length = len(normalize_text(p["content"]))
            segment = cleaned_text[current_pos:current_pos + length].strip()
            result_pages.append({
                "page": p["page"],
                "content": segment
            })
            current_pos += length
        return result_pages

    return pages_text


def normalize_text(text):
    return (
        text.replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\u202f", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )
