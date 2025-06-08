# Use Python 3.11 with more complete base image for better compatibility
FROM python:3.11-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    BUILD_VERSION=2025-01-06-v2

# Set work directory
WORKDIR /app

# Install comprehensive system dependencies for audio/video processing
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # Audio libraries
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    libportaudio2 \
    # Video/codec libraries  
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavdevice-dev \
    libavfilter-dev \
    # SSL and crypto
    libssl-dev \
    libffi-dev \
    # Additional dependencies
    libjpeg-dev \
    libopus-dev \
    libvpx-dev \
    libsrtp2-dev \
    # Utilities
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel build

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with WORKING VERSIONS
RUN echo "Installing compatible versions - Build $(date)" && \
    pip install --no-cache-dir fastapi==0.104.1 uvicorn[standard]==0.24.0 && \
    pip install --no-cache-dir google-genai>=0.8.0 && \
    pip install --no-cache-dir numpy==1.26.4 scipy==1.11.4 && \
    pip install --no-cache-dir python-dotenv pydantic pydantic-settings && \
    pip install --no-cache-dir httpx aiofiles python-multipart websockets && \
    pip install --no-cache-dir gunicorn psutil && \
    pip install --no-cache-dir av==10.0.0 && \
    pip install --no-cache-dir aiortc==1.4.0 && \
    pip install --no-cache-dir fastrtc==0.0.24

# Debug aiortc structure and test compatibility
RUN echo "=== Debugging aiortc structure ===" && \
    python -c "import aiortc; print('aiortc version:', getattr(aiortc, '__version__', 'unknown'))" && \
    python -c "import aiortc; print('aiortc dir:', [x for x in dir(aiortc) if not x.startswith('_')])" && \
    python -c "try: from aiortc.mediastreams import AudioStreamTrack; print('✅ AudioStreamTrack found in mediastreams'); except: print('❌ AudioStreamTrack not in mediastreams')" && \
    python -c "try: from aiortc import AudioStreamTrack; print('✅ AudioStreamTrack in main module'); except: print('❌ AudioStreamTrack not in main module')" && \
    echo "=== Testing FastRTC compatibility ===" && \
    python -c "import fastrtc; print('✅ FastRTC basic import works')" || echo "❌ FastRTC basic import failed"

# Verify ONLY critical imports (skip FastRTC components for now)
RUN python -c "import fastapi, uvicorn, google.genai; print('✅ Core imports OK')" && \
    python -c "import numpy, scipy, av; print('✅ Audio/video imports OK')" && \
    python -c "import aiortc; print('✅ aiortc import OK')" && \
    python -c "import fastrtc; print('✅ fastrtc import OK')"

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
