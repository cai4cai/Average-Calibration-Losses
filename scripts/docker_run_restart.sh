#!/bin/bash

# Usage example:
# ./docker_run_restart.sh --mode train --bundle acdc17_softl1ace_dice_ce_2_ace_10_0 --seed 12345 [--gpu <gpu_number>] [--cpus <cpus>] [--shm-size <shm_size>] [--max-restarts <number>]
#
# To run in background and disconnect terminal:
# nohup ./docker_run_restart.sh --mode train --bundle acdc17_softl1ace_dice_ce_2_ace_10_0 --seed 12345 --gpu 0 --cpus 16-23 > restart_log.txt 2>&1 &
#
# Or use the script directly with & and disown:
# ./docker_run_restart.sh --mode train --bundle acdc17_softl1ace_dice_ce_2_ace_10_0 --seed 12345 --gpu 0 --cpus 16-23 > restart_log.txt 2>&1 & disown

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$0")"

# Navigate to the parent directory of the script, which is assumed to be the project root
PROJECT_DIR="$(realpath "$SCRIPT_DIR/..")"

# Define the data directory relative to the project directory
# Assuming the data directory is at the same level as the SACROS project directory
DATA_DIR="$(realpath "$PROJECT_DIR/../data")"

# Docker image name
IMAGE_NAME="${USER}/acl:latest"

# Default values for optional arguments
GPU=0         # Default GPU index
CPUS="0-5"    # Default CPU range
SHM_SIZE="32g" # Default shared memory size
SEED=12345    # Default seed value
MAX_RESTARTS=1000  # Maximum number of restarts (set high to keep trying)
RESTART_DELAY=5   # Seconds to wait before restarting
BUNDLE=""     # Will be extracted from args

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPU="$2"
      shift # past argument
      shift # past value
      ;;
    --cpus)
      CPUS="$2"
      shift # past argument
      shift # past value
      ;;
    --shm-size)
      SHM_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --seed)
      SEED="$2"
      shift # past argument
      shift # past value
      ;;
    --max-restarts)
      MAX_RESTARTS="$2"
      shift # past argument
      shift # past value
      ;;
    --bundle)
      BUNDLE="$2"
      POSITIONAL_ARGS+=("$1")
      POSITIONAL_ARGS+=("$2")
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL_ARGS+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

# Create log directory if it doesn't exist
LOG_DIR="$PROJECT_DIR/restart_logs"
mkdir -p "$LOG_DIR"

# Create a log file with timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/restart_${BUNDLE}_seed${SEED}_${TIMESTAMP}.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_message "=========================================="
log_message "Auto-Restart Docker Training Script"
log_message "=========================================="
log_message "Using GPU: $GPU, CPUs: $CPUS, SHM_SIZE: $SHM_SIZE, SEED: $SEED"
log_message "Max restarts: $MAX_RESTARTS"
log_message "Arguments to run_monai_bundle.py: $@"
log_message "PROJECT_DIR: $PROJECT_DIR"
log_message "DATA_DIR: $DATA_DIR"
log_message "Log file: $LOG_FILE"
log_message "=========================================="

# Counter for restarts
RESTART_COUNT=0

# Function to run the docker container and wait for it
run_training() {
    # Start container in detached mode and capture container ID
    CONTAINER_ID=$(docker run -d --rm \
        --gpus '"device='$GPU'"' \
        --cpuset-cpus=$CPUS \
        --shm-size=$SHM_SIZE \
        --volume $PROJECT_DIR:/workspace/project \
        --volume $DATA_DIR:/workspace/data \
        --workdir /workspace/project \
        $IMAGE_NAME \
        python ./run_monai_bundle.py "$@" --seed $SEED)

    if [ -z "$CONTAINER_ID" ]; then
        log_message "ERROR: Failed to start container"
        return 1
    fi

    log_message "Started container: $CONTAINER_ID"

    # Wait for the container to finish and get exit code
    docker wait "$CONTAINER_ID" > /dev/null 2>&1
    EXIT_CODE=$?

    # If docker wait failed, try to inspect the container
    if [ $EXIT_CODE -ne 0 ]; then
        # Container may have already exited, try to get its exit code
        INSPECT_EXIT=$(docker inspect "$CONTAINER_ID" --format='{{.State.ExitCode}}' 2>/dev/null)
        if [ ! -z "$INSPECT_EXIT" ]; then
            EXIT_CODE=$INSPECT_EXIT
        fi
    fi

    return $EXIT_CODE
}

# Main restart loop
while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    RESTART_COUNT=$((RESTART_COUNT + 1))

    log_message ""
    log_message "=========================================="
    log_message "Starting training attempt #$RESTART_COUNT"
    log_message "=========================================="

    # Run the training and wait for completion
    run_training "$@"
    EXIT_CODE=$?

    log_message ""
    log_message "=========================================="
    log_message "Training exited with code: $EXIT_CODE"

    # If exit code is 0, training completed successfully
    if [ $EXIT_CODE -eq 0 ]; then
        log_message "Training completed successfully!"
        log_message "=========================================="
        exit 0
    fi

    # If we've reached max restarts, exit
    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        log_message "Reached maximum restart limit ($MAX_RESTARTS)"
        log_message "=========================================="
        exit 1
    fi

    # Otherwise, wait and restart
    log_message "Training crashed. Restarting in $RESTART_DELAY seconds..."
    log_message "=========================================="
    sleep $RESTART_DELAY
done

log_message "Restart loop completed"
exit 1
