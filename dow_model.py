from sentence_transformers import SentenceTransformer
import logging
import os

MODEL_PATH = "models"


def load_and_save_model():
    if not os.path.exists(MODEL_PATH):
        logging.info(f"Đang tải mô hình từ {MODEL_PATH}...")
        # Tải mô hình từ Hugging Face
        model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
        # Lưu mô hình vào thư mục cục bộ
        model.save(MODEL_PATH)
        logging.info(f"Mô hình đã được lưu tại {MODEL_PATH}")
    else:
        logging.info(f"Mô hình đã có sẵn tại {MODEL_PATH}, không cần tải lại.")
        # Tải mô hình từ thư mục cục bộ
        model = SentenceTransformer(MODEL_PATH)

    return model


model = load_and_save_model()
