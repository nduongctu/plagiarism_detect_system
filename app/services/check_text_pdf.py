import fitz
import io


def check_if_text_pdf(pdf_stream: io.BytesIO) -> bool:
    try:
        pdf_stream.seek(0)
        doc = fitz.open("pdf", pdf_stream.getvalue())
        for page in doc:
            text = page.get_text("text").strip()
            if text:
                return True
        return False
    except Exception:
        return False
