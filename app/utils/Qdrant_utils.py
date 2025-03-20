from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.config.settings import QDRANT_HOST, COLLECTION_NAME, VECTOR_SIZE, DISTANCE_METRIC

DISTANCE_MAPPING = {
    "cosine": Distance.COSINE,
    "euclidean": Distance.EUCLID,
    "dot": Distance.DOT
}

distance_metric = DISTANCE_MAPPING.get(DISTANCE_METRIC, Distance.COSINE)

client = QdrantClient(QDRANT_HOST)

#client.delete_collection(collection_name=COLLECTION_NAME)

# Kiểm tra sự tồn tại của collection
collections = client.get_collections().collections
existing_collection_names = [col.name for col in collections]

if COLLECTION_NAME not in existing_collection_names:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=distance_metric)
    )
