import fitz

def check_if_text_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return bool(text.strip())
    except Exception as e:
        return False
