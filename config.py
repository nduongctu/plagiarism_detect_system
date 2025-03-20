import torch

QDRANT_HOST = "http://qdrant:6333"
COLLECTION_NAME = "plagiarism_check"

PDF_FOLDER_PATH = "pdf"
VECTOR_SIZE = 1024
DISTANCE_METRIC = "cosine"

EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"

PDF_DIR = "pdf"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

CHUNK_SIZE = 256
CHUNK_OVERLAP = 0
MIN_CHUNK_LENGTH = 100

SIMILARITY_THRESHOLD = 0.8
