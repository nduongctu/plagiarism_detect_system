import time
import torch
from app.config import settings
from qdrant_client import QdrantClient, models
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.file_utils import clean_text, extract_text_without_headers_footers, process_chunks
from app.services.save_to_Qdrant import embedding_model
from io import BytesIO

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


def plagiarism_check(pdf_bytes: BytesIO, threshold: float, chunk_size: int, chunk_overlap: int, min_chunk_length: int):
    print("\nBắt đầu kiểm tra đạo văn...")

    start_total = time.time()

    print("\nĐang tải và xử lý file PDF...")
    start_step = time.time()
    query_chunks = split_text_into_chunks(pdf_bytes, chunk_size, chunk_overlap, min_chunk_length)
    print(f"-> Thời gian xử lý file PDF: {time.time() - start_step:.4f} s")

    total_chunks = len(query_chunks)
    print(f"Tổng số đoạn văn đã chia: {total_chunks}")

    print("\nĐang trích xuất embeddings...")
    start_step = time.time()
    query_texts = [chunk.page_content for chunk in query_chunks]
    query_embeddings = embed_texts(query_texts)
    print(f"-> Thời gian trích xuất embeddings: {time.time() - start_step:.4f} s")

    print("\nĐang kiểm tra với dữ liệu trong Qdrant...")
    start_step = time.time()
    client = QdrantClient(settings.QDRANT_HOST)
    matches = compare_with_qdrant(query_chunks, query_embeddings, client, threshold)
    print(f"-> Thời gian kiểm tra Qdrant: {time.time() - start_step:.4f} s")

    end_total = time.time()
    print(f"\nTổng thời gian kiểm tra đạo văn: {end_total - start_total:.4f} s")

    result = {
        "duplication_rate": 0,
        "message": "Phát hiện trùng lặp",
        "duplicate_passages": []
    }

    if matches:
        seen_query_chunks = set()
        unique_matches = []
        for match in matches:
            query_text = match['query_text']
            if query_text not in seen_query_chunks:
                seen_query_chunks.add(query_text)
                unique_matches.append(match)

        unique_match_count = len(unique_matches)
        duplication_rate = (unique_match_count / len(query_chunks)) * 100
        result["duplication_rate"] = format_duplication_rate(duplication_rate)

        for match in unique_matches:
            result["duplicate_passages"].append(match)
    else:
        result["message"] = "Không tìm thấy đoạn trùng lặp."

    return result
