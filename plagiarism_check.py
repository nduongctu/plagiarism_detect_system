import os
import numpy as np
import torch
import torch.nn.functional as F
import config
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from process_file import clean_text, extract_text_without_headers_footers, process_chunks
from process_to_qdrant import embedding_model

DEVICE = config.DEVICE


def split_text_into_chunks(pdf_path):
    pages_content = extract_text_without_headers_footers(pdf_path, skip_pages={0})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=30,
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
                    "source": pdf_path,
                    "page": page["page"],
                    "start": start_idx,
                    "end": end_idx
                })
                start_idx = end_idx - config.CHUNK_OVERLAP

            processed_chunks, processed_metadata = process_chunks(raw_chunks, page_metadata)

            for chunk, meta in zip(processed_chunks, processed_metadata):
                chunks.append(Document(page_content=chunk, metadata=meta))

    return chunks


def embed_texts(texts):
    model = embedding_model
    embeddings = []
    for i in range(0, len(texts), config.BATCH_SIZE):
        batch = texts[i:i + config.BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, device=DEVICE)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


def compare_with_qdrant(query_chunks, query_embeddings, client, threshold=config.SIMILARITY_THRESHOLD):
    search_queries = [
        models.QueryRequest(
            query=query_emb.tolist(),
            limit=3,
            with_payload=True
        )
        for query_emb in query_embeddings
    ]

    search_results = client.query_batch_points(
        collection_name=config.COLLECTION_NAME,
        requests=search_queries
    )

    matches = []

    for i, query_response in enumerate(search_results):
        if not hasattr(query_response, 'points'):
            continue

        for scored_point in query_response.points:
            try:
                # Access score attribute directly
                if hasattr(scored_point, 'score'):
                    score = scored_point.score
                else:
                    continue

                # Access payload attribute directly
                if hasattr(scored_point, 'payload'):
                    payload = scored_point.payload
                else:
                    continue

                if score >= threshold:
                    matches.append({
                        "query_text": query_chunks[i].page_content,
                        "matched_text": payload.get("content", "N/A") if isinstance(payload, dict) else "N/A",
                        "similarity": int(round(float(score) * 100)),
                        "source_file": payload.get("source", "unknown") if isinstance(payload, dict) else "unknown",
                        "source_page": payload.get("page", "?") if isinstance(payload, dict) else "?",
                        "start": payload.get("start", 0) if isinstance(payload, dict) else 0,
                        "end": payload.get("end", 0) if isinstance(payload, dict) else 0
                    })
            except Exception as e:
                print(f"Error processing hit: {e}")
                continue

    if not matches:
        return []

    return sorted(matches, key=lambda x: x["similarity"], reverse=True)


def format_duplication_rate(rate):
    return int(rate) if rate.is_integer() else round(rate, 2)


def plagiarism_check(pdf_path):
    print("\nĐang tải và xử lý file PDF...")
    query_chunks = split_text_into_chunks(pdf_path)
    total_chunks = len(query_chunks)
    print(f"Tổng số đoạn văn đã chia: {total_chunks}")

    print("\nĐang trích xuất embeddings...")
    query_texts = [chunk.page_content for chunk in query_chunks]
    query_embeddings = embed_texts(query_texts)

    print("\nĐang kiểm tra với dữ liệu trong Qdrant...")
    client = QdrantClient(config.QDRANT_HOST)
    matches = compare_with_qdrant(query_chunks, query_embeddings, client)

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

# result = plagiarism_check("uploads/lv_nguyendaiduong.pdf")
# print(result)
