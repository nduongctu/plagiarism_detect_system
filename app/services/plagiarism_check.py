import time
import torch
from io import BytesIO
from app.config import settings
from qdrant_client import QdrantClient, models
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.file_utils import clean_text, extract_text_without_headers_footers, process_chunks
from app.services.save_to_Qdrant import embedding_model
import re
from app.config import settings

DEVICE = settings.DEVICE


def split_text_into_chunks(pdf_bytes: BytesIO):
    start_time = time.time()

    pages_content = extract_text_without_headers_footers(pdf_bytes, skip_pages={0})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
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

            processed_chunks, processed_metadata = process_chunks(raw_chunks, page_metadata, settings.MIN_CHUNK_LENGTH)

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


def count_words_in_text(text: str) -> int:
    return len(text.split())


def calculate_word_plagiarism_rate(common_phrases, query_text):
    total_words_query = count_words_in_text(query_text)
    total_common_words = sum(phrase["length"] for phrase in common_phrases)
    duplication_rate_by_words = (total_common_words / total_words_query) * 100 if total_words_query > 0 else 0
    # Giới hạn không vượt quá 100%
    duplication_rate_by_words = min(duplication_rate_by_words, 100)
    return round(duplication_rate_by_words, 2)


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


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().split()


def find_common_phrases(query_text, matched_text, min_length):
    words1 = preprocess_text(query_text)
    words2 = preprocess_text(matched_text)

    common_phrases = []
    visited_query = set()
    visited_match = set()

    i = 0
    while i < len(words1):
        if i in visited_query:
            i += 1
            continue

        for j in range(len(words2)):
            if j in visited_match:
                continue

            if words1[i] == words2[j]:
                temp_phrase = [words1[i]]
                i_temp, j_temp = i + 1, j + 1

                while (
                        i_temp < len(words1) and j_temp < len(words2)
                        and words1[i_temp] == words2[j_temp]
                ):
                    temp_phrase.append(words1[i_temp])
                    visited_query.add(i_temp)
                    visited_match.add(j_temp)
                    i_temp += 1
                    j_temp += 1

                if len(temp_phrase) >= min_length:
                    common_phrases.append({
                        "phrase": " ".join(temp_phrase),
                        "query_index": i,
                        "matched_index": j,
                        "length": len(temp_phrase)
                    })
                    visited_query.update(range(i, i_temp))
                    visited_match.update(range(j, j_temp))
                    break

        i += 1

    return common_phrases, len(words2)


def classify_plagiarism(matches, x=55, n=2):
    classified_results = []

    for match in matches:
        query_text = match["query_text"]
        matched_text = match["matched_text"]

        common_phrases, total_matched_words = find_common_phrases(query_text, matched_text, n)

        total_common_words = sum(phrase["length"] for phrase in common_phrases)
        similarity_percentage = (total_common_words / total_matched_words) * 100 if total_matched_words > 0 else 0
        plagiarism_type = "Đạo từ ngữ" if similarity_percentage > x else "Đạo ngữ nghĩa"

        classified_results.append({
            **match,
            "plagiarism_type": plagiarism_type,
            "common_phrases": common_phrases,
        })

    return classified_results


def plagiarism_check(pdf_bytes: BytesIO, threshold: float = None, x: int = 55, n: int = 2):
    print("\nBắt đầu kiểm tra đạo văn...")

    start_total = time.time()

    if threshold is None:
        threshold = settings.SIMILARITY_THRESHOLD

    print("\nĐang tải và xử lý file PDF...")
    start_step = time.time()
    query_chunks = split_text_into_chunks(pdf_bytes)
    print(f"-> Thời gian xử lý file PDF: {time.time() - start_step:.4f} s")

    total_chunks = len(query_chunks)
    print(f"Tổng số đoạn văn đã chia: {total_chunks}")

    if total_chunks == 0:
        return {
            "duplication_rate": 0,
            "message": "Tài liệu quá ngắn hoặc không có nội dung hợp lệ để kiểm tra.",
            "duplicate_passages": []
        }

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
        "message": "Không tìm thấy đoạn trùng lặp.",
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

        classified_matches = classify_plagiarism(unique_matches, x, n)

        unique_match_count = len(unique_matches)
        duplication_rate = (unique_match_count / total_chunks) * 100 if total_chunks > 0 else 0

        total_common_rate = 0
        for match in classified_matches:
            total_common_rate += calculate_word_plagiarism_rate(match["common_phrases"], match["query_text"])
        avg_duplication_rate_by_words = total_common_rate / len(classified_matches) if classified_matches else 0

        result["duplication_rate"] = format_duplication_rate(duplication_rate)
        result["duplication_rate_by_words"] = avg_duplication_rate_by_words
        result["message"] = "Phát hiện trùng lặp"
        result["duplicate_passages"] = classified_matches

    return result
