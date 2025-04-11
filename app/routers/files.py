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

router = APIRouter()

uploaded_files = {}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split(".")[-1].lower()

        if file_ext not in ["pdf", "docx", "doc"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        file_bytes = await file.read()
        filename_without_ext = os.path.splitext(file.filename)[0]

        if file_ext == "pdf":
            uploaded_files["latest_file"] = {
                "filename": file.filename,
                "data": file_bytes,
            }

            return {
                "filename": file.filename,
                "message": "PDF saved in memory",
            }

        docx_stream = io.BytesIO(file_bytes)

        if file_ext == "doc":
            try:
                doc = Document(docx_stream)
                docx_stream = io.BytesIO()
                doc.save(docx_stream)
                docx_stream.seek(0)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error converting DOC to DOCX: {str(e)}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_docx:
            temp_docx.write(docx_stream.getvalue())
            temp_docx.flush()
            temp_docx_path = temp_docx.name

        pdf_path = temp_docx_path.replace(".docx", ".pdf")
        try:
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", os.path.dirname(pdf_path),
                 temp_docx_path],
                check=True
            )

            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Failed to convert DOCX to PDF: {str(e)}")

        finally:
            os.remove(temp_docx_path)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

        uploaded_files["latest_file"] = {
            "filename": f"{filename_without_ext}.pdf",
            "data": pdf_bytes,
        }

        return {
            "filename": f"{filename_without_ext}.pdf",
            "message": "Converted and saved in memory",
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

    file_info = uploaded_files["latest_file"]
    filename = file_info["filename"]
    pdf_stream = io.BytesIO(file_info["data"])
    pdf_stream.seek(0)

    threshold_value = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    result = plagiarism_check(pdf_stream, threshold=threshold_value, n=n)

    return {
        "filename": filename,
        "result": result
    }


@router.get("/extract_info")
async def extract_info():
    if "latest_file" not in uploaded_files:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        pdf_stream = io.BytesIO(uploaded_files["latest_file"]["data"])
        result = extract_info_with_gemini(pdf_stream)

        if isinstance(result, str) and result.startswith("{\"error\""):
            return JSONResponse(content=result, status_code=500)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi trích xuất thông tin: {e}")


@router.post("/save-file")
async def save_file():
    if "latest_file" not in uploaded_files:
        raise HTTPException(status_code=404, detail="No uploaded file in memory")

    file_info = uploaded_files["latest_file"]
    filename = file_info["filename"]
    pdf_stream = io.BytesIO(file_info["data"])
    pdf_stream.seek(0)

    result = save_uploaded_pdf(pdf_stream, filename)

    return {"message": result["message"]}
