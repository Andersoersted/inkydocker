# syntax=docker/dockerfile:1

# Build arguments
ARG USE_GPU=false
ARG PYTHON_VERSION=3.13.2

# ===== BUILDER STAGE =====
FROM python:${PYTHON_VERSION}-slim AS builder

# Set timezone and cache directory for models
ENV TZ=Europe/Copenhagen
ENV XDG_CACHE_HOME=/app/data/model_cache
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create necessary directories for the model cache
RUN mkdir -p /app/data/model_cache

# Pass the GPU flag to the builder stage
ARG USE_GPU

# Install build dependencies - this layer rarely changes
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Split requirements into base and model-specific for better caching
# Create base requirements file
RUN echo "Flask\n\
Flask-SQLAlchemy==3.1.1\n\
cryptography\n\
requests>=2.31.0\n\
pyppeteer>=1.0.2\n\
Flask-Migrate\n\
httpx\n\
APScheduler>=3.9.0\n\
celery\n\
redis\n\
tqdm>=4.66.1\n\
gunicorn\n\
gevent\n\
psutil>=5.9.0\n\
pytz>=2024.1\n\
Pillow>=11.0.0\n\
pillow-heif>=0.13.0" > base_requirements.txt

# Create model-specific requirements file
RUN echo "scikit-learn>=1.4.1.post1\n\
open_clip_torch>=2.20.0\n\
torch>=2.2.1\n\
torchvision>=0.17.1" > model_requirements.txt

# Install base requirements first (these change less frequently)
RUN pip install --upgrade pip && pip install --no-cache-dir -r base_requirements.txt

# Copy tasks.py for modification
COPY tasks.py .

# If CPU-only mode is selected, modify tasks.py to force CPU usage
RUN if [ "$USE_GPU" = "false" ]; then \
    echo "Building CPU-only version"; \
    sed -i 's/device = "cuda" if torch.cuda.is_available() else "cpu"/device = "cpu"  # Force CPU usage/g' tasks.py; \
    fi

# Install PyTorch and model-specific dependencies separately for better caching
RUN if [ "$USE_GPU" = "false" ]; then \
    # Install CPU-only PyTorch first
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    # Then install the model-specific requirements
    pip install --no-cache-dir -r model_requirements.txt; \
  else \
    # Install with default PyTorch (with CUDA)
    pip install --no-cache-dir torch torchvision torchaudio && \
    pip install --no-cache-dir -r model_requirements.txt; \
  fi

# Create model cache directory
RUN mkdir -p /build/model_cache

# Pre-download OpenCLIP models in a separate layer for better caching
# Set higher recursion limit to avoid "maximum recursion depth exceeded" errors
RUN python -c "import sys; import open_clip; \
    print('Setting recursion limit to 15000...'); \
    sys.setrecursionlimit(15000); \
    print('Downloading OpenCLIP ViT-B-32 model (small)...'); \
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); \
    print('OpenCLIP ViT-B-32 model downloaded successfully.')"

# Pre-download medium OpenCLIP model
RUN python -c "import sys; import open_clip; \
    print('Setting recursion limit to 15000...'); \
    sys.setrecursionlimit(15000); \
    print('Downloading OpenCLIP ViT-B-16 model (medium)...'); \
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai'); \
    print('OpenCLIP ViT-B-16 model downloaded successfully.')"

# Pre-download large OpenCLIP model with proper recursion limit
# Remove the fallback to ensure the model is actually downloaded
RUN python -c "import sys; import open_clip; \
    print('Setting recursion limit to 15000...'); \
    sys.setrecursionlimit(15000); \
    print('Downloading OpenCLIP ViT-L-14 model (large)...'); \
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai'); \
    print('OpenCLIP ViT-L-14 model downloaded successfully.')"

# Create directory for model files
RUN mkdir -p /app/data/models

# ===== FINAL STAGE =====
FROM python:${PYTHON_VERSION}-slim

# Pass the GPU flag to the final stage
ARG USE_GPU

# Set timezone and cache directory for models
ENV TZ=Europe/Copenhagen
ENV XDG_CACHE_HOME=/app/data/model_cache

# Install only runtime dependencies - this layer rarely changes
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    supervisor \
    tzdata \
    redis-server \
    bc \
    # Runtime libraries needed for Python packages
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create necessary directories for the database and model cache
RUN mkdir -p /data /app/data/model_cache /app/data/ram_models

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the model caches from the builder stage
COPY --from=builder /app/data/model_cache /app/data/model_cache
COPY --from=builder /app/data/models /app/data/models

# Copy the modified tasks.py from builder stage
COPY --from=builder /build/tasks.py /app/tasks.py

# Copy configuration files first (these change less frequently)
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy application files last (these change most frequently)
# This ensures that changes to application code don't invalidate the dependency cache
COPY . .

# Make scheduler.py executable
RUN chmod +x /app/scheduler.py

# Set environment variables for Celery (no memory limits)
ENV CELERY_WORKERS=2

# Expose port 5001
EXPOSE 5001

# Tell Flask which file is our app
ENV FLASK_APP=app.py

# Run entrypoint.sh (which handles migrations and launches Supervisor)
CMD ["/entrypoint.sh"]