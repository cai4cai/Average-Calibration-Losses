#!/bin/bash

# Arguments from batch_submit.py
bundle=$1
mode=$2
seed=$3
gpu_memory=${4:-"32G"}  # Default to 32G if not provided
cpu_memory=${5:-"64G"}  # Default to 64G if not provided
job_name=$6

# Calculate memory limit as 1.5 times the cpu_memory
memory_limit=$(echo "${cpu_memory}" | awk '{printf "%.0fG\n", $1 * 1.5}')

# Delete existing job with the same name if exists
runai delete job tb-"${job_name}"

# Submit new job with the specified parameters
runai submit tb-"${job_name}" \
  -i aicregistry:5000/${USER}/sacros \
  -p tbarfoot \
  --gpu-memory "${gpu_memory}" \
  --node-type "dgx1" \
  --cpu "16" \
  --cpu-limit "20" \
  --memory "${cpu_memory}" \
  --memory-limit "${memory_limit}" \
  --host-ipc \
  -v /nfs:/nfs \
  -e MPLCONFIGDIR="/tmp/" \
  -- bash /nfs/home/${USER}/SACROS/scripts_runai/startup.sh "$bundle" "$mode" "$seed"

  #  -g 1 \
  #  --node-type "A100" \
  # --node-type "dgx1" \
