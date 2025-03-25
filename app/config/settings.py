import torch

QDRANT_HOST = "http://qdrant:6333"
COLLECTION_NAME = "plagiarism_check"

PDF_FOLDER_PATH = "../../data/pdf"
VECTOR_SIZE = 768
DISTANCE_METRIC = "cosine"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

CHUNK_SIZE = 156
CHUNK_OVERLAP = 25
MIN_CHUNK_LENGTH = 100
SIMILARITY_THRESHOLD = 0.7

CHUNK_SIZE_opt1 = 80
CHUNK_OVERLAP_opt1 = 15
MIN_CHUNK_LENGTH_opt1 = 50
SIMILARITY_THRESHOLD_opt1 = 0.83
