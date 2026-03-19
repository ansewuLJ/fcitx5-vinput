# Qwen3-ASR Server Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
    fastapi uvicorn \
    librosa soundfile \
    numpy \
    websockets

# Install qwen-asr package
RUN pip install --no-cache-dir qwen-asr

# Copy server code
COPY qwen_asr_server.py /app/

# Create directory for models
RUN mkdir -p /models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "qwen_asr_server.py", "--model", "Qwen/Qwen3-ASR-0.6B", "--host", "0.0.0.0", "--port", "8000"]
