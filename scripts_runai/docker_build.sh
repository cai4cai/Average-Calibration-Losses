#!/bin/bash
# Create a "tag" or name for the image
docker_tag=aicregistry:5000/${USER}/sacros

CUDA_VERSION="12.4.0"
PYTORCH_VERSION="2.5.0"
CUDATOOLKIT_VERSION="12.4"
PYTHON_VERSION="3.12"

docker build ../docker -f ../docker/Dockerfile \
 --tag "${docker_tag}" \
 --build-arg USER_ID="$(id -u)" \
 --build-arg GROUP_ID="$(id -g)" \
 --build-arg USER="${USER}" \
 --build-arg CUDA_VERSION="${CUDA_VERSION}" \
 --build-arg PYTORCH_VERSION="${PYTORCH_VERSION}" \
 --build-arg CUDATOOLKIT_VERSION="${CUDATOOLKIT_VERSION}" \
 --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
 --network=host

docker push "${docker_tag}"