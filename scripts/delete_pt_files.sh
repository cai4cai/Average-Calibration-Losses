#!/bin/bash

# Move to the directory where the script is located
cd "$(dirname "$0")"

# Define the project root directory relative to this script location
# Assuming the script is in the root/subfolder, adjust the path accordingly
PROJECT_ROOT="$(git rev-parse --show-toplevel)"

# Define source directory
SRC_DIR="$PROJECT_ROOT/bundles/"

echo "Source directory: $SRC_DIR"

# Find all .pt files, print their names, and delete them
find "$SRC_DIR" -name '*.pt' -exec echo "Deleting {}" \; -delete
