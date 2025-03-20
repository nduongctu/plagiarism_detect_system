FROM python:3.10-slim

RUN apt-get update

RUN pip install typing-extensions==4.10.0
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]