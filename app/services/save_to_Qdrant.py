import os
import io
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
from app.config.settings import PDF_FOLDER_PATH, DEVICE, CHUNK_SIZE, COLLECTION_NAME, \
    CHUNK_OVERLAP, BATCH_SIZE
from app.utils.Qdrant_utils import client
from app.utils.file_utils import clean_text, extract_text_without_headers_footers, process_chunks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models/DEk21_hcmute_embedding")
embedding_model = SentenceTransformer(MODEL_PATH, device=DEVICE)


def save_uploaded_pdf(pdf_stream: io.BytesIO, filename: str):
    print(f"Đang xử lý file: {filename}")

    pages_content = extract_text_without_headers_footers(pdf_stream)
    documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks = []
    metadata_list = []

    for page_data in pages_content:
        cleaned_text = clean_text(page_data["content"])
        if cleaned_text:
            raw_chunks = text_splitter.split_text(cleaned_text)

            start_pos = 0
            for chunk in raw_chunks:
                end_pos = start_pos + len(chunk)
                metadata_list.append({
                    "start": start_pos,
                    "end": end_pos,
                    "page": page_data["page"],
                    "source": filename
                })
                all_chunks.append(chunk)
                start_pos = end_pos

    if all_chunks:
        processed_chunks, processed_metadata = process_chunks(all_chunks, metadata_list)

        # Batch encoding
        chunk_vectors = embedding_model.encode(processed_chunks, batch_size=BATCH_SIZE, show_progress_bar=True)

        documents = [
            PointStruct(
                id=hash(chunk) % (10 ** 9),
                vector=vector.tolist(),
                payload={
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "content": chunk,
                    "start": metadata["start"],
                    "end": metadata["end"]
                }
            )
            for chunk, vector, metadata in zip(processed_chunks, chunk_vectors, processed_metadata)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=documents)
        print(f"{filename} đã lưu thành công với {len(documents)} đoạn văn bản!")
        return {"message": f"{filename} đã lưu thành công với {len(documents)} đoạn văn bản!"}

    print(f"{filename} không có nội dung hợp lệ để lưu.")
    return {"message": f"{filename} không có nội dung hợp lệ để lưu."}


def process_all_pdfs_in_folder(folder_path=PDF_FOLDER_PATH):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print("Không có file PDF nào trong thư mục!")
        return {"message": "Không có file PDF nào trong thư mục!"}

    for filename in pdf_files:
        pdf_path = os.path.join(folder_path, filename)
        with open(pdf_path, "rb") as pdf_stream:
            save_uploaded_pdf(pdf_stream, filename)

# process_all_pdfs_in_folder()
