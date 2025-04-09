import io
import os
import json
import subprocess
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from docx import Document
from app.services.plagiarism_check import plagiarism_check
from app.services.save_to_Qdrant import save_uploaded_pdf
from app.config import settings
from app.services.extract_info import extract_info_with_gemini
from app.services.check_text_pdf import check_if_text_pdf

router = APIRouter()

uploaded_files = {}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split(".")[-1].lower()

        if file_ext not in ["pdf", "docx", "doc"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        filename_without_ext = os.path.splitext(file.filename)[0]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf_path = temp_pdf.name

            if file_ext == "pdf":
                pdf_bytes = await file.read()
                with open(temp_pdf_path, "wb") as temp_pdf_file:
                    temp_pdf_file.write(pdf_bytes)

                is_text_pdf = check_if_text_pdf(temp_pdf_path)

                # Lưu trạng thái PDF vào uploaded_files
                uploaded_files["latest_file"] = {
                    "filename": f"{filename_without_ext}.pdf",
                    "path": temp_pdf_path,
                    "is_text_pdf": is_text_pdf
                }

            else:
                # Xử lý file DOCX hoặc DOC
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_docx:
                    docx_temp_path = temp_docx.name
                    with open(docx_temp_path, "wb") as temp_docx_file:
                        temp_docx_file.write(await file.read())

                    try:
                        # Chuyển đổi DOCX sang PDF
                        subprocess.run(
                            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir",
                             os.path.dirname(temp_pdf_path),
                             docx_temp_path],
                            check=True
                        )
                        converted_pdf_path = docx_temp_path.replace(".docx", ".pdf")
                        with open(converted_pdf_path, "rb") as converted_pdf:
                            with open(temp_pdf_path, "wb") as temp_pdf_file:
                                temp_pdf_file.write(converted_pdf.read())

                    except subprocess.CalledProcessError as e:
                        raise HTTPException(status_code=500, detail=f"Failed to convert DOCX to PDF: {str(e)}")
                    finally:
                        os.remove(docx_temp_path)
                        if os.path.exists(converted_pdf_path):
                            os.remove(converted_pdf_path)

                # Lưu thông tin vào uploaded_files
                uploaded_files["latest_file"] = {
                    "filename": f"{filename_without_ext}.pdf",
                    "path": temp_pdf_path,
                    "is_text_pdf": True
                }

        # Trích xuất metadata từ file PDF và Gemini API
        metadata = extract_info_with_gemini(uploaded_files["latest_file"]["path"],
                                            uploaded_files["latest_file"]["is_text_pdf"])

        return {
            "filename": f"{filename_without_ext}.pdf",
            "message": "File saved to disk, metadata extracted.",
            "metadata": metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


@router.post("/check-plagiarism")
async def check_plagiarism(
        threshold: float = Query(0.75, description="Threshold for plagiarism check (default 0.75)"),
        n: int = Query(2, description="Minimum length of common phrase (default 2)")
):
    if "latest_file" not in uploaded_files:
        raise HTTPException(status_code=404, detail="No uploaded file found")

    filename = uploaded_files["latest_file"]["filename"]
    pdf_stream = io.BytesIO(uploaded_files["latest_file"]["data"])
    pdf_stream.seek(0)

    threshold_value = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    result = plagiarism_check(pdf_stream, threshold=threshold_value, n=n)

    return {"filename": filename, "result": result}


@router.post("/save-file")
async def save_file():
    if "latest_file" not in uploaded_files:
        raise HTTPException(status_code=404, detail="No uploaded file in memory")

    filename = uploaded_files["latest_file"]["filename"]
    pdf_stream = io.BytesIO(uploaded_files["latest_file"]["data"])
    pdf_stream.seek(0)

    result = save_uploaded_pdf(pdf_stream, filename)

    return {"message": result["message"]}
