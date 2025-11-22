FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-lg.txt

COPY . .

EXPOSE 8005
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]