from fastapi import APIRouter
import os
from plagiarism_check import plagiarism_check
from process_to_qdrant import save_uploaded_pdf
router = APIRouter()

UPLOAD_DIR = "uploads"


@router.post("/check-plagiarism")
async def check_plagiarism():
    if not os.path.exists(UPLOAD_DIR):
        raise HTTPException(status_code=404, detail="Uploads directory not found")

    results = []
    for file_name in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".pdf"):
            result = plagiarism_check(file_path)
            results.append({"file_name": file_name, "result": result})

    if not results:
        return {"message": "No PDF files found in uploads directory"}

    return results


@router.post("/save-file")
async def save_file():
    if not os.path.exists(UPLOAD_DIR):
        raise HTTPException(status_code=404, detail="Uploads directory not found")

    messages = []
    for file_name in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".pdf"):
            result = save_uploaded_pdf(file_path)
            messages.append(result["message"])

    return {"messages": messages}
