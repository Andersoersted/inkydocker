# syntax=docker/dockerfile:1

# Set default build arguments
ARG BASE_IMAGE=python:3.13.2-slim
FROM ${BASE_IMAGE}

# Set timezone and cache directory for models (persisted in /data/model_cache)
ENV TZ=Europe/Copenhagen
ENV XDG_CACHE_HOME=/app/data/model_cache

# Install system dependencies and redis-server
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    supervisor \
    tzdata \
    build-essential \
    gcc \
    git \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    redis-server \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories for the database and model cache
RUN mkdir -p /data /app/data/model_cache

# Set working directory
WORKDIR /app

# Copy only the requirements file first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Pre-download all CLIP models (this layer will be cached if requirements.txt hasn't changed)
RUN python -c "import open_clip; \
    open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', jit=False, force_quick_gelu=True); \
    open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', jit=False, force_quick_gelu=True); \
    open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', jit=False, force_quick_gelu=True)"

# Copy the rest of the project files
COPY . .

# Make scheduler.py executable
RUN chmod +x /app/scheduler.py

# Set environment variables for Celery
ENV CELERY_WORKER_MAX_MEMORY_PER_CHILD=500000
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