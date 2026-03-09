FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    faster-whisper \
    flask \
    coqui-tts \
    pydub \
    diarize \
    "transformers>=4.57,<5.0"

WORKDIR /app
COPY app.py .
COPY transcribe.py .
COPY static/ static/

EXPOSE 5000

CMD ["python", "app.py"]
