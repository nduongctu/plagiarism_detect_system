import torch

QDRANT_HOST = "http://qdrant:6333"
COLLECTION_NAME = "plagiarism_check"

PDF_FOLDER_PATH = "../../data/pdf"
VECTOR_SIZE = 768
DISTANCE_METRIC = "cosine"

EMBEDDING_MODEL = "huyydangg/DEk21_hcmute_embedding"

PDF_DIR = "../../data/pdf"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

CHUNK_SIZE = 128
CHUNK_OVERLAP = 0
MIN_CHUNK_LENGTH = 50

SIMILARITY_THRESHOLD = 0.75
