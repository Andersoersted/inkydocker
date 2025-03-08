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

# Install build dependencies
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

# Copy requirements and tasks.py for modification
COPY requirements.txt .
COPY tasks.py .

# If CPU-only mode is selected, modify tasks.py to force CPU usage
RUN if [ "$USE_GPU" = "false" ]; then \
    echo "Building CPU-only version"; \
    sed -i 's/device = "cuda" if torch.cuda.is_available() else "cpu"/device = "cpu"  # Force CPU usage/g' tasks.py; \
    fi

# Install Python dependencies (with CPU-only PyTorch if USE_GPU=false)
RUN if [ "$USE_GPU" = "false" ]; then \
    # Install CPU-only PyTorch first
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    # Then install the rest of the requirements
    pip install --no-cache-dir -r requirements.txt; \
  else \
    # Install with default PyTorch (with CUDA)
    pip install --no-cache-dir -r requirements.txt; \
  fi

# Pre-download only the smallest CLIP model to ensure it's available in the image
# Other models can be downloaded on-demand by the user
RUN python -c "import open_clip; \
    print('Downloading ViT-B-32 model...'); \
    open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True); \
    print('Model downloaded successfully.')"

# ===== FINAL STAGE =====
FROM python:${PYTHON_VERSION}-slim

# Pass the GPU flag to the final stage
ARG USE_GPU

# Set timezone and cache directory for models
ENV TZ=Europe/Copenhagen
ENV XDG_CACHE_HOME=/app/data/model_cache

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    supervisor \
    tzdata \
    redis-server \
    # Runtime libraries needed for Python packages
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create necessary directories for the database and model cache
RUN mkdir -p /data /app/data/model_cache

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Only the smallest CLIP model (ViT-B-32) is pre-downloaded in the builder stage
# and copied with the site-packages. Other models can be downloaded on-demand by the user
# through the settings page, which provides a more flexible approach while keeping the
# Docker image size smaller.

# Copy the modified tasks.py from builder stage
COPY --from=builder /build/tasks.py /app/tasks.py

# Copy application files (except tasks.py which was already copied)
COPY . .

# Make scheduler.py executable
RUN chmod +x /app/scheduler.py

# Set environment variables for Celery (no memory limits)
ENV CELERY_WORKERS=2

# Expose port 5001
EXPOSE 5001

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy Supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Tell Flask which file is our app
ENV FLASK_APP=app.py

# Run entrypoint.sh (which handles migrations and launches Supervisor)
CMD ["/entrypoint.sh"]