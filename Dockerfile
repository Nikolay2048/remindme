FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg for audio/video, build tools for some wheels
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Базовые зависимости ----
COPY requirements.base.txt .
RUN pip install --no-cache-dir -r requirements.base.txt

# ---- Torch-зависимости ----
ARG TORCH_REQUIREMENTS=requirements.gpu.txt
COPY ${TORCH_REQUIREMENTS} ./torch-requirements.txt
RUN pip install --no-cache-dir -r torch-requirements.txt

RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY src/ ./src/
COPY main.py ./main.py

CMD ["python", "main.py"]