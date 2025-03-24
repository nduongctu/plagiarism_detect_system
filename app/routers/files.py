import io
import os
import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from docx import Document
from docx2pdf import convert
from app.services.plagiarism_check import plagiarism_check
from app.services.save_to_Qdrant import save_uploaded_pdf
from app.config import settings

router = APIRouter()

uploaded_files = {}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split(".")[-1].lower()

        if file_ext not in ["pdf", "docx", "doc"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        file_bytes = await file.read()

        if file_ext in ["docx", "doc"]:
            docx_stream = io.BytesIO(file_bytes)

            if file_ext == "doc":
                try:
                    doc = Document(docx_stream)
                    docx_stream = io.BytesIO()
                    doc.save(docx_stream)
                    docx_stream.seek(0)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error converting DOC to DOCX: {str(e)}")

            pdf_stream = io.BytesIO()
            convert(docx_stream, pdf_stream)
            pdf_stream.seek(0)
            file_bytes = pdf_stream.getvalue()

        uploaded_files["latest_file"] = {"filename": file.filename, "data": file_bytes}

        return {"filename": file.filename, "message": "File processed in memory"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


@router.post("/check-plagiarism")
async def check_plagiarism(option: int = Query(1, ge=1, le=2), threshold: float = Query(None)):
    if "latest_file" not in uploaded_files:
        raise HTTPException(status_code=404, detail="No uploaded file found")

    filename = uploaded_files["latest_file"]["filename"]
    pdf_stream = io.BytesIO(uploaded_files["latest_file"]["data"])
    pdf_stream.seek(0)

    if option == 1:
        chunk_size = settings.CHUNK_SIZE_opt1
        chunk_overlap = settings.CHUNK_OVERLAP_opt1
        min_chunk_length = settings.MIN_CHUNK_LENGTH_opt1
        similarity_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD_opt1
    else:
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        min_chunk_length = settings.MIN_CHUNK_LENGTH
        similarity_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    # Gọi hàm kiểm tra đạo văn với cấu hình đã chọn
    result = plagiarism_check(pdf_stream, threshold=similarity_threshold, chunk_size=chunk_size,
                              chunk_overlap=chunk_overlap, min_chunk_length=min_chunk_length)

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
