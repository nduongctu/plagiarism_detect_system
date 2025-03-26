import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models/Llama-3.2-1B-en-vi")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)


def classify_plagiarism_with_genai(matches):
    """
    Kiểm tra đạo văn sau khi lấy kết quả từ Qdrant.
    Chỉ giữ lại những kết quả "trùng từ ngữ" hoặc "trùng ngữ nghĩa".
    """
    classified_results = []

    for match in matches:
        query_text = match["query_text"]
        matched_text = match["matched_text"]

        prompt = f"""
        Hãy kiểm tra xem đoạn văn bản đầu tiên (query_text) có sao chép hoặc diễn đạt lại theo nghĩa tương tự trong đoạn văn bản thứ hai (matched_text) hay không.
        - Nếu các câu trùng lặp về từ ngữ (câu chữ giống nhau hoặc gần giống), hãy đánh dấu là "trùng từ ngữ".
        - Nếu các câu không giống nhau về câu chữ nhưng truyền tải ý nghĩa tương tự, hãy đánh dấu là "trùng ngữ nghĩa".
        - Nếu không có sự trùng lặp đáng kể, hãy đánh dấu là "không trùng".
        Trả về chỉ một giá trị: "trùng từ ngữ", "trùng ngữ nghĩa" hoặc "không trùng".
        query_text: "{query_text}"
        matched_text: "{matched_text}"
        """

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, temperature=0.2)

        classification = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()

        if classification in ["trùng từ ngữ", "trùng ngữ nghĩa"]:
            match["loai_dao_van"] = classification
            classified_results.append(match)

    return classified_results
