#!/bin/bash
# Script to build and push the InkyDocker image to Docker Hub

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or not installed"
  exit 1
fi

# Get Docker Hub username if not provided
if [ -z "$1" ]; then
  read -p "Enter your Docker Hub username: " DOCKER_USERNAME
else
  DOCKER_USERNAME=$1
fi

# Get version tag if not provided
if [ -z "$2" ]; then
  read -p "Enter version tag (default: latest): " IMAGE_TAG
  IMAGE_TAG=${IMAGE_TAG:-latest}
else
  IMAGE_TAG=$2
fi

# Ask if GPU support should be enabled
if [ -z "$3" ]; then
  read -p "Enable GPU support? (y/N): " GPU_SUPPORT
  if [[ "$GPU_SUPPORT" =~ ^[Yy]$ ]]; then
    USE_GPU=true
    TAG_SUFFIX="-gpu"
  else
    USE_GPU=false
    TAG_SUFFIX=""
  fi
else
  if [ "$3" = "gpu" ]; then
    USE_GPU=true
    TAG_SUFFIX="-gpu"
  else
    USE_GPU=false
    TAG_SUFFIX=""
  fi
fi

echo "Building and pushing image as $DOCKER_USERNAME/inkydocker:$IMAGE_TAG$TAG_SUFFIX"
echo "GPU support: $USE_GPU"

# Build the image with the specified tag and GPU setting
DOCKER_USERNAME=$DOCKER_USERNAME IMAGE_TAG=$IMAGE_TAG USE_GPU=$USE_GPU docker-compose build

# Check if build was successful
if [ $? -ne 0 ]; then
  echo "Error: Docker build failed"
  exit 1
fi

# Log in to Docker Hub
echo "Logging in to Docker Hub..."
docker login

# Check if login was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to log in to Docker Hub"
  exit 1
fi

# Push the image
echo "Pushing image to Docker Hub..."
docker push $DOCKER_USERNAME/inkydocker:$IMAGE_TAG$TAG_SUFFIX

# Check if push was successful
if [ $? -eq 0 ]; then
  echo "Success! Image pushed to Docker Hub as $DOCKER_USERNAME/inkydocker:$IMAGE_TAG$TAG_SUFFIX"
else
  echo "Error: Failed to push image to Docker Hub"
  exit 1
fi

# If the tag is not 'latest', also tag and push as latest with appropriate suffix
if [ "$IMAGE_TAG" != "latest" ]; then
  echo "Also tagging and pushing as 'latest$TAG_SUFFIX'..."
  docker tag $DOCKER_USERNAME/inkydocker:$IMAGE_TAG$TAG_SUFFIX $DOCKER_USERNAME/inkydocker:latest$TAG_SUFFIX
  docker push $DOCKER_USERNAME/inkydocker:latest$TAG_SUFFIX
fi

echo "Done!"