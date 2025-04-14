import os
import fitz
import re
import json
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def extract_info_with_gemini(pdf_stream):
    try:
        pdf_bytes = pdf_stream.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        text = ""
        for i in range(min(4, doc.page_count)):
            page_text = doc[i].get_text("text").strip()
            text += page_text + "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        model = genai.GenerativeModel("gemini-2.0-flash")

        if text and len(text) >= 15:
            prompt = f"""
Phân tích văn bản sau và trích xuất thông tin theo định dạng JSON, kết quả trả về tiếng Việt.

Dữ liệu đầu vào:
{text}

Yêu cầu đầu ra:
{{
    "title": "string // Tên đề tài nghiên cứu hoặc sáng kiến, nếu có thể xác định.",
    "author": [ 
        {{
            "name": "string // Họ và tên đầy đủ của tác giả.",
            "gender": "string // Giới tính của tác giả, có thể dựa theo tên (male/female/None).",
            "dob": "string // Ngày sinh của tác giả theo định dạng YYYY-MM-DD, nếu có.",
            "email": "string // Địa chỉ email của tác giả, nếu có.",
            "phone": "string // Số điện thoại của tác giả, nếu có.",
            "organization": "string // Tên tổ chức/tên đơn vị (bệnh viện,...) mà tác giả trực thuộc.",
            "department": "string // Tên khoa/phòng ban của tác giả trong tổ chức/đơn vị (nếu có).",
            "position": "string // Chức vụ/chức danh của tác giả tại tổ chức/đơn vị (nếu có)."
        }}
    ],
    "type": "string // Loại hồ sơ (chỉ trả về Sáng kiến hoặc Đề tài KH&CN, đề án khoa học).",
    "proposal_level_city": "string // Đề nghị công nhận phạm vi ảnh hưởng : có đánh dấu ở ô Cấp thành phố(trả về true hoặc false).",
    "research_field": "string // Lĩnh vực nghiên cứu chính của bài viết, nếu có thể xác định."
}}
"""
            response = model.generate_content(prompt)

        else:
            new_doc = fitz.open()
            for i in range(min(5, doc.page_count)):
                new_doc.insert_pdf(doc, from_page=i, to_page=i)

            new_pdf_stream = BytesIO()
            new_doc.save(new_pdf_stream)
            new_doc.close()
            new_pdf_stream.seek(0)

            uploaded_file = genai.upload_file(new_pdf_stream, mime_type="application/pdf", display_name="scan.pdf")

            prompt = """
Đây là file PDF chứa 5 trang đầu của một tài liệu scan. Hãy phân tích nội dung file và trích xuất thông tin theo định dạng JSON, kết quả trả về tiếng Việt.

Yêu cầu đầu ra:
{
    "title": "string // Tên đề tài nghiên cứu hoặc sáng kiến, nếu có thể xác định.",
    "author": [ 
        {
            "name": "string // Họ và tên đầy đủ của tác giả.",
            "gender": "string // Giới tính của tác giả, có thể dựa theo tên (male/female/None).",
            "dob": "string // Ngày sinh của tác giả theo định dạng YYYY-MM-DD, nếu có.",
            "email": "string // Địa chỉ email của tác giả, nếu có.",
            "phone": "string // Số điện thoại của tác giả, nếu có.",
            "organization": "string // Tên tổ chức/tên đơn vị (bệnh viện,...) mà tác giả trực thuộc.",
            "department": "string // Tên khoa/phòng ban của tác giả trong tổ chức/đơn vị (nếu có).",
            "position": "string // Chức vụ/chức danh của tác giả tại tổ chức/đơn vị (nếu có)."
        }
    ],
    "type": "string // Loại hồ sơ (chỉ trả về Sáng kiến hoặc Đề tài KH&CN, đề án khoa học).",
    "proposal_level_city": "string // Đề nghị công nhận phạm vi ảnh hưởng : có đánh dấu ở ô Cấp thành phố(trả về true hoặc false).",
    "research_field": "string // Lĩnh vực nghiên cứu chính của bài viết, nếu có thể xác định."
}
"""
            response = model.generate_content([prompt, uploaded_file])

        doc.close()

        content = response.text.strip().lstrip('\ufeff')
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {"error": f"Lỗi JSON: {str(e)}"}

    except Exception as e:
        return {"error": f"Lỗi xử lý văn bản: {str(e)}"}
