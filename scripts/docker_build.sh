#!/bin/bash

# Create a "tag" or name for the image
docker_tag=${USER}/acl:latest

docker build --no-cache ../docker -f ../docker/Dockerfile \
 --tag "${docker_tag}" \
 --build-arg USER_ID="$(id -u)" \
 --build-arg GROUP_ID="$(id -g)" \
 --build-arg USER="${USER}" \
 --network=host

# docker push "${docker_tag}"
