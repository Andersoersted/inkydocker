# InkyDocker

A Docker-based application for managing and sending images to e-ink displays.

## Multi-Stage Docker Build

This project uses a multi-stage Docker build to create a smaller, more efficient Docker image. The build process is split into two stages:

1. **Builder Stage**: Installs all build dependencies and compiles/installs Python packages
2. **Final Stage**: Contains only the runtime dependencies and copies the built packages from the builder stage

## Key Features

- Optimized Docker image size through multi-stage build
- Redis included in the main container for simplicity
- Improved caching of Docker layers
- Reduced CLIP model footprint (only pre-downloads the smallest model)

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Building and Running

#### CPU-only Build (Default)

To build and run the application with CPU-only support:

```bash
docker-compose up --build
```

#### GPU-enabled Build

To build with GPU support (requires CUDA-compatible GPU and drivers):

```bash
USE_GPU=true docker-compose up --build
```

Both options will:
1. Build the Docker image using the multi-stage Dockerfile
2. Start the application container with Redis included
3. Mount the data and images directories as volumes

### Choosing Between CPU and GPU

- **CPU Build**: Smaller image size, works on any machine, suitable for most use cases
  - Explicitly installs CPU-only versions of PyTorch and related libraries
  - Forces CPU usage in the code regardless of CUDA availability
  - No CUDA dependencies are included in the image
- **GPU Build**: Faster AI processing for image tagging, requires CUDA-compatible hardware
  - Includes CUDA dependencies for GPU acceleration

### Pushing to Docker Hub

There are two ways to push your image to Docker Hub:

#### Option 1: Using the provided script

Use the included script which handles building, tagging, and pushing in one step:

```bash
./push-to-dockerhub.sh [username] [version] [gpu|cpu]
```

For example:
```bash
# CPU version (default)
./push-to-dockerhub.sh myusername 1.0.0

# GPU version
./push-to-dockerhub.sh myusername 1.0.0 gpu
```

If you don't provide parameters, the script will prompt you for them.

#### Option 2: Manual process

1. **Log in to Docker Hub**:
   ```bash
   docker login
   ```

2. **Build with environment variables**:
   ```bash
   # CPU version
   DOCKER_USERNAME=yourusername IMAGE_TAG=1.0.0 docker-compose build

   # GPU version
   DOCKER_USERNAME=yourusername IMAGE_TAG=1.0.0 USE_GPU=true docker-compose build
   ```

3. **Push the image**:
   ```bash
   # CPU version
   docker push yourusername/inkydocker:1.0.0

   # GPU version
   docker push yourusername/inkydocker:1.0.0-gpu
   ```

4. **Also push as latest** (optional):
   ```bash
   # CPU version
   docker tag yourusername/inkydocker:1.0.0 yourusername/inkydocker:latest
   docker push yourusername/inkydocker:latest

   # GPU version
   docker tag yourusername/inkydocker:1.0.0-gpu yourusername/inkydocker:latest-gpu
   docker push yourusername/inkydocker:latest-gpu
   ```

### Using Docker Buildx (Recommended for Unraid)

For Unraid users or anyone who wants to build and push directly to Docker Hub, you can use Docker Buildx:

#### For CPU-only build (default):
```bash
# This will explicitly install CPU-only versions of PyTorch and related libraries
# No CUDA dependencies will be included in the image
docker buildx build --platform linux/amd64 \
  --build-arg USE_GPU=false \
  -t yourusername/inkydocker:cpu \
  --push .
```

#### For GPU-enabled build:
```bash
# This will include CUDA dependencies for GPU acceleration
docker buildx build --platform linux/amd64 \
  --build-arg USE_GPU=true \
  -t yourusername/inkydocker:gpu \
  --push .
```

### Configuration

The application is configured through environment variables in the docker-compose.yml file:

- `TZ`: Timezone (default: Europe/Copenhagen)

## Architecture

The application consists of several components:

- **Flask Web Application**: Serves the web interface and API
- **Celery Workers**: Process background tasks like image tagging
- **Redis**: Used as message broker for Celery and for caching (runs within the same container)
- **Scheduler**: Manages scheduled tasks like sending images to devices

## Volumes

The application uses two main volumes:

- `./data:/app/data`: Stores application data, including the SQLite database and model cache
- `./images:/app/images`: Stores the images managed by the application

## Optimizations

Several optimizations have been made to reduce the Docker image size:

1. **Multi-stage build**: Separates build dependencies from runtime dependencies
2. **Reduced CLIP models**: Only pre-downloads the smallest CLIP model (ViT-B-32)
3. **Integrated Redis**: Redis runs in the same container for simplicity
4. **Optimized cleanup**: Removes unnecessary files after installation
5. **.dockerignore**: Excludes unnecessary files from the build context

## Troubleshooting

If you encounter issues with the application:

1. Check the logs: `docker-compose logs`
2. Verify Redis is running inside the container: `docker-compose exec inkydocker redis-cli ping`
3. Verify the volumes are mounted correctly: `docker-compose config`