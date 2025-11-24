FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    libsndfile1 \
    python3-pip \
    python3.10 \
    python3.10-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./
COPY kokoro ./kokoro

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY . .

RUN useradd --create-home --home-dir /home/appuser appuser && chown -R appuser /app
USER appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8005
CMD ["python", "-m", "uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8005"]