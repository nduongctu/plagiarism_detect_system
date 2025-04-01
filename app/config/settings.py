import torch

QDRANT_HOST = "http://qdrant:6333"
COLLECTION_NAME = "plagiarism_check_v1"

PDF_FOLDER_PATH = "../../data/pdf"
VECTOR_SIZE = 768
DISTANCE_METRIC = "cosine"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

model_name = "gemini-2.0-flash-thinking-exp-01-21"

CHUNK_SIZE = 156
CHUNK_OVERLAP = 0
MIN_CHUNK_LENGTH = 100
SIMILARITY_THRESHOLD = 0.75
