import os
import re
import json
import fitz
from dotenv import load_dotenv
import google.generativeai as genai
from app.config import settings

load_dotenv()

model_name = settings.model_name
genai.configure(api_key=os.getenv('API_GEMINI_KEY'))
model = genai.GenerativeModel(model_name)


def extract_text_from_pdf_stream(pdf_stream):
    try:
        pdf_bytes = pdf_stream.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        num_pages = min(2, doc.page_count)
        full_text = ""
        for i in range(num_pages):
            full_text += doc[i].get_text("text") + "\n"

        doc.close()

        full_text = re.sub(r'\s+', ' ', full_text).strip()
        return full_text
    except Exception as e:
        return f"Lỗi: {e}"


def extract_info_with_gemini(pdf_stream):
    pdf_text = extract_text_from_pdf_stream(pdf_stream)

    prompt = f"""
    Phân tích văn bản sau và trích xuất thông tin theo định dạng JSON, kết quả trả về tiếng Việt.
    Dữ liệu đầu vào:
    {pdf_text}

    Yêu cầu đầu ra:
    {{
        "title": "string // Tiêu đề của bài viết, nếu có thể xác định.",
        "author": [
            {{
                "name": "string // Họ và tên đầy đủ của tác giả.",
                "gender": "string // Giới tính của tác giả, có thể dựa theo tên (male/female/null).",
                "dob": "string // Ngày sinh của tác giả theo định dạng YYYY-MM-DD, nếu có.",
                "email": "string // Địa chỉ email của tác giả, nếu có.",
                "phone": "string // Số điện thoại của tác giả, nếu có.",
                "organization": "string // Tên tổ chức (trường đại học, viện nghiên cứu, công ty,...) mà tác giả trực thuộc.",
                "department": "string // Tên phòng ban của tác giả trong tổ chức (nếu có).",
                "position": "string // Chức vụ của tác giả tại tổ chức (nếu có)."
            }}
        ],
        "research_field": "string // Lĩnh vực nghiên cứu chính của bài viết, nếu có thể xác định."
    }}
    """

    try:
        response = model.generate_content(prompt)

        if response and response.text:
            response_text = response.text.replace('"unknown"', 'null').strip().lstrip('\ufeff')

            response_text = re.sub(r"^```json\s*", "", response_text)
            response_text = re.sub(r"\s*```$", "", response_text)

            if not response_text:
                return '{"error": "Response text is empty after processing."}'

            try:
                json_data = json.loads(response_text)
                return json_data
            except json.JSONDecodeError as e:
                return f'{{"error": "Lỗi JSON: {str(e)}"}}'
        else:
            return '{"error": "Không có phản hồi từ API Gemini."}'

    except Exception as e:
        return f'{{"error": "Lỗi API: {e}"}}'
