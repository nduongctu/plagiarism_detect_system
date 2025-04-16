import re
import fitz
import easyocr
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes
from app.config.settings import MIN_CHUNK_LENGTH


def clean_text_with_mapping(text):
    original_text = text
    text = re.sub(r"[\n\r]+", " ", text)

    # Khởi tạo mapping mới
    mapping = []
    cleaned_chars = []

    # Tạo mảng mapping và chuỗi cleaned_text ban đầu
    index_in_original = 0
    for char in text:
        if char.isalnum() or char.isspace() or char in ".[]":
            mapping.append(index_in_original)
            cleaned_chars.append(char)
        index_in_original += 1
    cleaned_text = "".join(cleaned_chars)

    # Tạo bản sao để sử dụng trong các phép thay thế
    temp_text = cleaned_text

    # 2. Xoá ngày tháng năm (viết bằng số hoặc chữ)
    temp_text = re.sub(
        r"\bngày\s+[^\s]+\s+tháng\s+[^\s]+\s+năm\s+\d{4}\b",
        " ",
        temp_text,
        flags=re.IGNORECASE
    )

    # 3. Xoá các cụm "Mục tiêu nghiên cứu", "Thiết kế nghiên cứu", "Dân số chọn mẫu"
    temp_text = re.sub(
        r"\b(mục\s*tiêu\s*nghiên\s*cứu|thiết\s*kế\s*nghiên\s*cứu|dân\s*số\s*chọn\s*mẫu)\b",
        " ",
        temp_text,
        flags=re.IGNORECASE
    )

    # 4. Xoá các cụm hành chính
    temp_text = re.sub(r"\bkính\s*gửi\b", " ", temp_text, flags=re.IGNORECASE)
    temp_text = re.sub(
        r"\bhội\s*đồng\s*xét\s*công\s*nhận\s*sáng\s*kiến\s*cấp\s*thành\s*phố\b",
        " ",
        temp_text,
        flags=re.IGNORECASE
    )
    temp_text = re.sub(r"\bsở\s*y\s*tế\b", " ", temp_text, flags=re.IGNORECASE)
    temp_text = re.sub(r"\bhội\s*đồng\s*xét\s*công\s*nhận\s*sở\s*y\s*tế\s*tp\.?\s*hcm\b", " ", temp_text,
                       flags=re.IGNORECASE)
    temp_text = re.sub(r"\bhội\s*đồng\s*xét\s*công\s*nhận\s*sáng\s*kiến\s*tp\.?\s*hcm\b", " ", temp_text,
                       flags=re.IGNORECASE)
    temp_text = re.sub(r"\bBỆNH\s+VIỆN\b", " ", temp_text)

    # 5. Xoá dấu gạch nối và gạch dưới
    temp_text = re.sub(r"(?:\s*-\s*)+", " ", temp_text)
    temp_text = re.sub(r"_", " ", temp_text)

    # 6. Xoá các chỉ mục dạng [số] (không có khoảng trắng)
    temp_text = re.sub(r"\[\d+\]", " ", temp_text)

    # 7. Xoá ký tự đặc biệt
    temp_text = re.sub(r"(\d)[,.](\d)", r"\1#DECIMAL#\2", temp_text)  # Thay thế tạm thời dấu thập phân
    temp_text = re.sub(r"[^\w\s#%]", " ", temp_text)  # Xóa các ký tự đặc biệt khác
    temp_text = re.sub(r"#DECIMAL#", ".", temp_text)  # Khôi phục dấu chấm

    # 8. Xoá các cụm khẩu hiệu
    temp_text = re.sub(r"\bCỘNG\s*HOÀ\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM\b", " ", temp_text,
                       flags=re.IGNORECASE)
    temp_text = re.sub(r"\bTHÀNH\s*PHÓ\s*HỎ\s*CHÍ\s*MINH\b", " ", temp_text, flags=re.IGNORECASE)
    temp_text = re.sub(r"\bThành\s*phố\s*Hô\s*Chí\s*Minh\b", " ", temp_text, flags=re.IGNORECASE)
    temp_text = re.sub(r"\bĐộc\s*lập\s*Tự\s*do\s*Hạnh\s*phúc\b", " ", temp_text, flags=re.IGNORECASE)

    # 9. Rút gọn khoảng trắng sau mỗi bước xử lý
    temp_text = re.sub(r"\s+", " ", temp_text).strip()

    # 10. Xoá cụm liên quan đến nghiệm thu và mô tả đề tài
    temp_text = re.sub(r"\bđã\s+được\s+nghi[êe]m\s+thu\b", " ", temp_text, flags=re.IGNORECASE)
    temp_text = re.sub(r"tên\s*đề\s*tà[i1]\s*nghiên\s*cứu", " ", temp_text, flags=re.IGNORECASE)
    temp_text = re.sub(
        r"mô\s*tả\s*tóm\s*tắt\s*nội\s*dung\s*của\s*đề\s*tà[i1]?\s*nghiên\s*cứu(\s*khoa\s*học)?(\s*chữ)?", " ",
        temp_text, flags=re.IGNORECASE)
    temp_text = re.sub(r"200\s*500\s*chữ", " ", temp_text)

    # 11. Loại bỏ các cụm tiêu đề của sáng kiến:
    temp_text = re.sub(
        r"m[\s\W_]*[ôoó][\s\W_]*t[\s\W_]*[ảaá][\s\W_]*t[\s\W_]*[óoô][\s\W_]*m[\s\W_]*t[\s\W_]*[ắaăạ][\s\W_]*t[\s\W_]*n[\s\W_]*[ộoó][\s\W_]*i[\s\W_]*d[\s\W_]*[uư][\s\W_]*n[\s\W_]*[gq][\s\W_]*c[\s\W_]*[ủuư][\s\W_]*a[\s\W_]*s[\s\W_]*[áa][\s\W_]*n[\s\W_]*[gq][\s\W_]*k[\s\W_]*[iíì][\s\W_]*[êeế][\s\W_]*[nñ]",
        " ",
        temp_text,
        flags=re.IGNORECASE
    )
    temp_text = re.sub(
        r"b[\s\W_]*[ôoố][\s\W_]*[iíì][\s\W_]*c[\s\W_]*[ảaá][\s\W_]*n[\s\W_]*h[\s\W_]*d[\s\W_]*[ẫăãâầẩẫ][\s\W_]*n[\s\W_]*t[\s\W_]*[ớơờở][\s\W_]*i[\s\W_]*s[\s\W_]*[áa][\s\W_]*n[\s\W_]*[gq][\s\W_]*k[\s\W_]*[iíì][\s\W_]*[êeế][\s\W_]*[nñ]",
        " ",
        temp_text,
        flags=re.IGNORECASE
    )
    temp_text = re.sub(
        r"\bMẪU\s+SỐ\b",
        " ",
        temp_text
    )
    temp_text = re.sub(
        r"t[eê]n\s*s[aá]ng\s*ki[eế]n",
        " ",
        temp_text,
        flags=re.IGNORECASE
    )
    temp_text = re.sub(
        r"n[oộ]i\s*dung\s*s[aá]ng\s*ki[eế]n",
        " ",
        temp_text,
        flags=re.IGNORECASE
    )
    temp_text = re.sub(
        r"m[oô]\s*t[aả]\s*k[ỹy]\s*thu[aậ]t",
        " ",
        temp_text,
        flags=re.IGNORECASE
    )

    # Cuối cùng rút gọn khoảng trắng
    temp_text = re.sub(r"\s+", " ", temp_text).strip()

    return temp_text, mapping


def normalize_text(text, preserve_length=False):
    if preserve_length:
        # Thay thế các ký tự đặc biệt bằng khoảng trắng, giữ nguyên độ dài
        text = text.replace("\xa0", " ")  # No-break space
        text = text.replace("\u200b", " ")  # Zero-width space
        text = text.replace("\u202f", " ")  # Narrow no-break space
        text = text.replace("\n", " ")  # Line feed
        text = text.replace("\r", " ")  # Carriage return
    else:
        # Loại bỏ các ký tự đặc biệt, có thể thay đổi độ dài
        text = text.replace("\xa0", " ")
        text = text.replace("\u200b", "")  # Loại bỏ hoàn toàn zero-width space
        text = text.replace("\u202f", " ")
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")

    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_signature_tail(pages_text: list) -> list:
    # Kiểm tra nếu đầu vào là chuỗi (string), chuyển đổi thành định dạng danh sách từ điển
    if isinstance(pages_text, str):
        # Tạo một đối tượng kiểu Dictionary với cấu trúc phù hợp
        pages_text = [{"page": 1, "content": pages_text}]

    # Các pattern xác định phần văn bản không cần thiết (ví dụ: phần xác nhận, tóm tắt)
    patterns = [r"XÁC NHẬN CỦA", r"NGƯỜI YÊU CẦU CÔNG NHẬN"]
    stop_pattern = r"TÓM\s*TẮT"

    # Kết hợp nội dung của tất cả các trang thành một chuỗi để kiểm tra
    # Dùng preserve_length=True để giữ nguyên độ dài khi tính toán vị trí
    page_lengths = []
    normalized_contents = []

    for p in pages_text:
        normalized = normalize_text(p["content"], preserve_length=True)
        normalized_contents.append(normalized)
        page_lengths.append(len(normalized))

    combined_text = " ".join(normalized_contents)

    start_idx = None
    for pattern in patterns:
        match = re.search(pattern, combined_text, flags=re.IGNORECASE)
        if match:
            start_idx = match.start()
            break

    if start_idx is None:
        return pages_text  # Không có văn bản cần xóa, trả về kết quả ban đầu

    # Xác định vị trí bắt đầu của trang cuối cùng
    last_page_start = sum(page_lengths[:-1]) + (len(pages_text) - 1)  # Cộng thêm số khoảng trắng giữa các trang

    # Xử lý khi mẫu xác nhận được tìm thấy
    if start_idx >= last_page_start:
        # Nếu mẫu tìm thấy trên trang cuối, cắt từ vị trí này trở đi
        end_idx = len(combined_text)
    else:
        # Nếu mẫu tìm thấy ở giữa các trang, tìm đến "TÓM TẮT"
        stop_match = re.search(stop_pattern, combined_text[start_idx:], flags=re.IGNORECASE)
        if stop_match:
            end_idx = start_idx + stop_match.start()
        else:
            end_idx = len(combined_text)

    # Cắt nội dung đã lọc, giữ lại phần trước start_idx
    cleaned_text = combined_text[:start_idx]

    # Nếu có phần kết thúc (sau end_idx), thêm vào
    if end_idx < len(combined_text):
        cleaned_text += combined_text[end_idx:]

    # Phân trang lại với việc giữ lại số trang
    result_pages = []
    current_pos = 0
    space_pos = 0  # Vị trí của khoảng trắng giữa các trang

    for i, p in enumerate(pages_text):
        if i > 0:
            space_pos += 1  # Mỗi lần chuyển trang, thêm 1 cho khoảng trắng giữa các trang

        # Tính toán vị trí hiện tại
        if current_pos >= len(cleaned_text):
            # Nếu đã hết nội dung làm sạch, thêm trang trống
            result_pages.append({
                "page": p["page"],
                "content": ""
            })
            continue

        # Tính độ dài tối đa có thể lấy từ cleaned_text
        page_length = page_lengths[i]
        end_pos = min(current_pos + page_length, len(cleaned_text))

        # Lấy nội dung trang từ văn bản đã làm sạch
        segment = cleaned_text[current_pos:end_pos].strip()

        # Thêm vào kết quả
        result_pages.append({
            "page": p["page"],
            "content": segment
        })

        # Cập nhật vị trí hiện tại cho trang tiếp theo
        current_pos = end_pos

    return result_pages


def extract_main_text_from_pdf(pdf_bytes: BytesIO, skip_pages=None):
    if skip_pages is None:
        skip_pages = set()

    raw_bytes = pdf_bytes.getvalue()
    text_blocks = []

    try:
        doc = fitz.open("pdf", raw_bytes)
        test_text = " ".join(doc[i].get_text("text").strip() for i in range(min(2, len(doc))))
        test_text = re.sub(r'\s+', ' ', test_text).strip()

        if not test_text or len(test_text) < 15:
            raise Exception("Không phát hiện đủ văn bản, chuyển qua OCR")

        stop_page = None
        for i in reversed(range(len(doc))):
            if re.search(r"tài liệu tham khảo", doc[i].get_text("text"), flags=re.IGNORECASE):
                stop_page = i + 1
                print(f"Dừng xử lý tại trang {stop_page} do phát hiện 'Tài liệu tham khảo'")
                break

        for i, page in enumerate(doc):
            page_index = i + 1
            if page_index in skip_pages:
                continue

            height = page.rect.height
            header_margin = height * (0.20 if i == 0 else 0.02)
            footer_margin = height * 0.02
            main_rect = fitz.Rect(page.rect.x0, page.rect.y0 + header_margin,
                                  page.rect.x1, page.rect.y1 - footer_margin)

            page_text = page.get_text("text", clip=main_rect).strip()
            if stop_page:
                if page_index == stop_page:
                    match = re.search(r"tài liệu tham khảo", page_text, flags=re.IGNORECASE)
                    if match:
                        page_text = page_text[:match.start()].strip()
                elif page_index > stop_page:
                    continue

            text_blocks.append({"page": page_index, "content": page_text})

        doc.close()

    except Exception as e:
        print(f"Fall back OCR do: {e}")

        images = convert_from_bytes(raw_bytes, dpi=350)

        reader = easyocr.Reader(['vi'], gpu=True)

        stop_index = None
        appendix_text = None
        appendix_page = None
        text_blocks = []

        for i, img in enumerate(images):
            page_index = i + 1
            if page_index in skip_pages:
                continue

            if i == 0:
                width, height = img.size
                img = img.crop((0, int(height * 0.20), width, height))

            img = img.resize((int(img.width * 2000 / max(img.size)), int(img.height * 2000 / max(img.size))),
                             Image.Resampling.LANCZOS
                             )

            img_array = np.array(img)
            del img

            text = reader.readtext(img_array, detail=0, paragraph=True, batch_size=15)

            del img_array

            text = " ".join(text).strip()

            if i == len(images) - 1 and re.search(r"PHỤ LỤC", text, flags=re.IGNORECASE):
                appendix_text = text
                appendix_page = page_index
                continue

            if stop_index is None:
                match = re.search(r"TÀI LIỆU THAM KHẢO", text, flags=re.IGNORECASE)
                if match:
                    text = text[:match.start()].strip()
                    stop_index = i
                    print(f"Phát hiện 'TÀI LIỆU THAM KHẢO' tại trang {page_index}, dừng sau trang này.")

            if stop_index is not None and i > stop_index:
                continue

            text_blocks.append({"page": page_index, "content": text})

        if appendix_text and appendix_page not in [b["page"] for b in text_blocks]:
            text_blocks.append({"page": appendix_page, "content": appendix_text})

    text_blocks = remove_signature_tail(text_blocks)
    text_blocks = remove_irrelevant_section(text_blocks)
    return text_blocks


def process_chunks(chunks, metadata_list, min_chunk_length=None):
    processed_chunks = []
    processed_metadata = []

    if min_chunk_length is None:
        min_chunk_length = MIN_CHUNK_LENGTH

    for chunk, metadata in zip(chunks, metadata_list):
        # Kiểm tra chất lượng chunk trước khi xử lý
        chunk = chunk.strip()

        if not chunk:  # Bỏ qua các chunk rỗng
            continue

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

    # Kết hợp nội dung với bảo tồn độ dài để xác định vị trí chính xác
    page_lengths = []
    normalized_contents = []

    for p in pages_text:
        normalized = normalize_text(p["content"], preserve_length=True)
        normalized_contents.append(normalized)
        page_lengths.append(len(normalized))

    combined_text = " ".join(normalized_contents)

    start_idx = None
    for pattern in start_patterns:
        match = re.search(pattern, combined_text, flags=re.IGNORECASE)
        if match:
            start_idx = match.start()
            break

    end_idx = None
    # Kiểm tra phần kết thúc
    if start_idx is not None:
        end_match = re.search(end_pattern, combined_text[start_idx:], flags=re.IGNORECASE)
        if end_match:
            end_idx = start_idx + end_match.end()

    if start_idx is not None and end_idx is not None:
        # Cắt nội dung đã lọc, giữ lại phần trước start_idx và sau end_idx
        cleaned_text = combined_text[:start_idx] + combined_text[end_idx:]

        # Phân trang lại với việc giữ lại số trang
        result_pages = []
        current_pos = 0
        space_pos = 0

        for i, p in enumerate(pages_text):
            if i > 0:
                space_pos += 1

            if current_pos >= len(cleaned_text):
                result_pages.append({
                    "page": p["page"],
                    "content": ""
                })
                continue

            page_length = page_lengths[i]
            end_pos = min(current_pos + page_length, len(cleaned_text))

            segment = cleaned_text[current_pos:end_pos].strip()

            result_pages.append({
                "page": p["page"],
                "content": segment
            })

            current_pos = end_pos

        return result_pages

    return pages_text
