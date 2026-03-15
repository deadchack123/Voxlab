FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: Heavy ML packages (rarely change)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    faster-whisper \
    coqui-tts \
    diarize \
    "transformers>=4.57,<5.0"

# Layer 2: Light/frequently updated packages
RUN pip install --no-cache-dir \
    flask \
    pydub \
    "yt-dlp[default]" \
    bgutil-ytdlp-pot-provider

WORKDIR /app
COPY app.py .
COPY transcribe.py .
COPY static/ static/

EXPOSE 5000

CMD ["python", "app.py"]
