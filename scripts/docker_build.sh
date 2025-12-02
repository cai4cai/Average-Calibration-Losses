#!/bin/bash

# Create a "tag" or name for the image
docker_tag=${USER}/acl:latest

docker build ../docker -f ../docker/Dockerfile \
 --tag "${docker_tag}" \
 --build-arg USER_UID="$(id -u)" \
 --build-arg USER_GID="$(id -g)" \
 --build-arg USERNAME="${USER}" \
 --network=host

# docker push "${docker_tag}"
#  --no-cache \