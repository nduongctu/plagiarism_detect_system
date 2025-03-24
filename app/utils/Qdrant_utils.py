from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
from app.config.settings import QDRANT_HOST, COLLECTION_NAME, VECTOR_SIZE, DISTANCE_METRIC

DISTANCE_MAPPING = {
    "cosine": Distance.COSINE,
    "euclidean": Distance.EUCLID,
    "dot": Distance.DOT
}
distance_metric = DISTANCE_MAPPING.get(DISTANCE_METRIC, Distance.COSINE)

# Kết nối Qdrant
client = QdrantClient(QDRANT_HOST)

collections = client.get_collections().collections
existing_collection_names = [col.name for col in collections]

if COLLECTION_NAME not in existing_collection_names:
    print(f"Tạo collection {COLLECTION_NAME} với HNSW index...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=distance_metric
        )
    )

    # Cấu hình HNSW sau khi tạo
    client.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=200
        )
    )
    print(f"Collection {COLLECTION_NAME} đã được tạo với HNSW index.")
else:
    print(f"Collection {COLLECTION_NAME} đã tồn tại.")
