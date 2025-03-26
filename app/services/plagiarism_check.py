import time
import torch
from io import BytesIO
from app.config import settings
from langchain.schema import Document
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.save_to_Qdrant import embedding_model
from app.services.classify_plagiarism import classify_plagiarism_with_genai
from app.utils.file_utils import clean_text, extract_text_without_headers_footers, process_chunks

DEVICE = settings.DEVICE


def split_text_into_chunks(pdf_bytes: BytesIO, chunk_size: int, chunk_overlap: int, min_chunk_length: int):
    start_time = time.time()

    pages_content = extract_text_without_headers_footers(pdf_bytes, skip_pages={0})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = []
    metadata_list = []

    for page in pages_content:
        cleaned_text = clean_text(page["content"])
        if cleaned_text:
            raw_chunks = text_splitter.split_text(cleaned_text)

            start_idx = 0
            page_metadata = []
            for chunk in raw_chunks:
                end_idx = start_idx + len(chunk)
                page_metadata.append({
                    "source": "uploaded_file",
                    "page": page["page"],
                    "start": start_idx,
                    "end": end_idx
                })
                start_idx = (end_idx - 20)

            processed_chunks, processed_metadata = process_chunks(raw_chunks, page_metadata, min_chunk_length)

            for chunk, meta in zip(processed_chunks, processed_metadata):
                chunks.append(Document(page_content=chunk, metadata=meta))

    end_time = time.time()
    print(f"Thời gian chia văn bản thành {len(chunks)} đoạn: {end_time - start_time:.4f} s")

    return chunks


def embed_texts(texts):
    model = embedding_model
    embeddings = []

    start_total = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), settings.BATCH_SIZE):
            batch = texts[i:i + settings.BATCH_SIZE]

            start_batch = time.time()
            batch_embeddings = model.encode(batch, convert_to_tensor=True, device=DEVICE)
            end_batch = time.time()

            print(
                f"Embedding batch {i // settings.BATCH_SIZE + 1}: {len(batch)} texts -> {end_batch - start_batch:.4f} s")

            embeddings.append(batch_embeddings)

    end_total = time.time()
    print(f"Tổng thời gian trích xuất embeddings: {end_total - start_total:.4f} s")

    return torch.cat(embeddings, dim=0)


def compare_with_qdrant(query_chunks, query_embeddings, client, threshold=settings.SIMILARITY_THRESHOLD):
    start_time = time.time()

    search_queries = [
        models.QueryRequest(
            query=query_emb.tolist(),
            limit=1,
            with_payload=True
        )
        for query_emb in query_embeddings
    ]

    search_results = client.query_batch_points(
        collection_name=settings.COLLECTION_NAME,
        requests=search_queries
    )

    end_time = time.time()
    print(f"Thời gian truy vấn Qdrant: {end_time - start_time:.4f} s")

    matches = []

    for i, query_response in enumerate(search_results):
        if not hasattr(query_response, 'points') or not query_response.points:
            continue

        scored_point = query_response.points[0]
        score = getattr(scored_point, "score", 0)
        payload = getattr(scored_point, "payload", {})

        if score >= threshold:
            matches.append({
                "query_text": query_chunks[i].page_content,
                "matched_text": payload.get("content", "N/A"),
                "similarity": int(round(float(score) * 100)),
                "source_file": payload.get("source", "unknown"),
                "source_page": payload.get("page", "?"),
                "start": payload.get("start", 0),
                "end": payload.get("end", 0)
            })

    return matches


def format_duplication_rate(rate):
    return int(rate) if rate.is_integer() else round(rate, 2)


def plagiarism_check(pdf_bytes: BytesIO, filename: str, threshold: float, chunk_size: int, chunk_overlap: int,
                     min_chunk_length: int):
    """
    Kiểm tra đạo văn bằng cách kết hợp Qdrant (cosine similarity) và GenAI (phân loại).
    """
    print("\nBắt đầu kiểm tra đạo văn...")

    start_total = time.time()

    print("\nĐang tải và xử lý file PDF...")
    query_chunks = split_text_into_chunks(pdf_bytes, chunk_size, chunk_overlap, min_chunk_length)

    print("\nĐang trích xuất embeddings...")
    query_texts = [chunk.page_content for chunk in query_chunks]
    query_embeddings = embed_texts(query_texts)

    print("\nĐang kiểm tra với dữ liệu trong Qdrant...")
    client = QdrantClient(settings.QDRANT_HOST)
    matches = compare_with_qdrant(query_chunks, query_embeddings, client, threshold)

    print("\nĐang phân loại đạo văn...")
    classified_matches = classify_plagiarism_with_genai(matches)

    total_words = sum(len(chunk.page_content.split()) for chunk in query_chunks)

    # Tính số từ bị trùng lặp
    duplicated_words = sum(len(match["query_text"].split()) for match in classified_matches)

    # Tính tỷ lệ trùng lặp theo số từ
    duplication_rate = (duplicated_words / total_words) * 100 if total_words > 0 else 0

    end_total = time.time()
    print(f"\nTổng thời gian kiểm tra đạo văn: {end_total - start_total:.4f} s")

    result = {
        "filename": filename,
        "result": {
            "duplication_rate": int(duplication_rate) if duplication_rate.is_integer() else round(duplication_rate, 2),
            "message": "Phát hiện trùng lặp" if classified_matches else "Không tìm thấy đoạn trùng lặp.",
            "duplicate_passages": classified_matches
        }
    }

    return result
