#!/bin/bash

# 1. Start the first job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle kits23_baseline_dice_ce_8 --seed 12345 --cpus "15-23" --gpu 1
docker wait "$(docker ps --latest --quiet)"

# 2. Start the second job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle kits23_hardl1ace_dice_ce_8 --seed 12345 --cpus "15-23" --gpu 1
docker wait "$(docker ps --latest --quiet)"

# 3. Start the third job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle kits23_softl1ace_dice_ce_8 --seed 12345 --cpus "15-23" --gpu 1
docker wait "$(docker ps --latest --quiet)"

# 1. Start the first job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle brats21_baseline_dice_ce_nl --seed 12345 --cpus "15-23" --gpu 1
docker wait "$(docker ps --latest --quiet)"

# 2. Start the second job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle brats21_hardl1ace_dice_ce_nl --seed 12345 --cpus "15-23" --gpu 1
docker wait "$(docker ps --latest --quiet)"

# 3. Start the third job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle brats21_softl1ace_dice_ce_nl --seed 12345 --cpus "15-23" --gpu 1
docker wait "$(docker ps --latest --quiet)"