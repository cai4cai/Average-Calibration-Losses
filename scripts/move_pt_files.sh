#!/bin/bash

# Move to the directory where the script is located
cd "$(dirname "$0")"

# Define the project root directory relative to this script location
# Assuming the script is in the root/subfolder, adjust the path accordingly
PROJECT_ROOT="$(git rev-parse --show-toplevel)"

# Define source and destination directories
SRC_DIR="$PROJECT_ROOT/bundles/"

# Get current date in YYYYMMDD format
TODAY=$(date +%Y%m%d)

# Define destination directory with current date
DEST_DIR="$PROJECT_ROOT/runs_pt_$TODAY/"

echo "Source directory: $SRC_DIR"
echo "Destination directory: $DEST_DIR"

# Create the destination directory if it does not exist
mkdir -p "$DEST_DIR"

# Copy all .pt files, preserving directory structure
rsync -avm --include='*.pt' --include='*/' --exclude='*' "$SRC_DIR" "$DEST_DIR"

# Check if rsync was successful
if [ $? -eq 0 ]; then
    echo "Copy successful. Deleting source files."
    # Find all .pt files and delete them from the source after successful copy
    find "$SRC_DIR" -name '*.pt' -exec echo "Deleting {}" \; -delete
else
    echo "Error: Copy failed. Source files not deleted."
fi
