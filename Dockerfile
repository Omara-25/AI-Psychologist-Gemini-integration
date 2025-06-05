# Use Python 3.11 slim image for better performance
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies in specific order to avoid conflicts
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir fastapi==0.104.1 uvicorn[standard]==0.24.0 && \
    pip install --no-cache-dir google-genai>=0.8.0 && \
    pip install --no-cache-dir numpy==1.26.4 scipy==1.11.4 && \
    pip install --no-cache-dir python-dotenv pydantic pydantic-settings && \
    pip install --no-cache-dir httpx aiofiles python-multipart websockets && \
    pip install --no-cache-dir gunicorn psutil && \
    pip install --no-cache-dir av==10.0.0 && \
    pip install --no-cache-dir aiortc==1.5.0 && \
    pip install --no-cache-dir fastrtc==0.0.24

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["python", "main.py"]
