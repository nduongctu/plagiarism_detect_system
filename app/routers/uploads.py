import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from docx import Document
from docx2pdf import convert

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Delete all existing files in the uploads directory
    for existing_file in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, existing_file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    if file.filename.endswith(".pdf"):
        return {"filename": file.filename, "file_path": file_path}
    elif file.filename.endswith(".doc") or file.filename.endswith(".docx"):
        docx_path = file_path
        if file.filename.endswith(".doc"):
            docx_path = file_path.replace(".doc", ".docx")
            # Convert DOC to DOCX
            doc = Document(file_path)
            doc.save(docx_path)
            os.remove(file_path)  # Delete the original .doc file

        pdf_path = docx_path.replace(".docx", ".pdf")

        convert(docx_path, pdf_path)
        os.remove(docx_path)  # Delete the .docx file after conversion
        return {"filename": os.path.basename(pdf_path), "file_path": pdf_path}
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
