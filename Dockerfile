FROM python:3.12-slim

ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    MODEL_CACHE_DIR=/cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers

WORKDIR /app

COPY requirements.lock .

RUN pip install --no-cache-dir --extra-index-url ${TORCH_CUDA_INDEX_URL} torch
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

RUN mkdir -p /app/data/kb /app/data/chroma /app/data/manifest /app/logs /cache/huggingface /cache/sentence-transformers

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
