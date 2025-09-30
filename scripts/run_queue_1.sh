#!/bin/bash

# 1. Start the first job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_baseline_ce_1 --seed 12345 --cpus "8-15" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 2. Start the second job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_hardl1ace_ce_1 --seed 12345 --cpus "8-15" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 3. Start the third job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_softl1ace_ce_1 --seed 12345 --cpus "8-15" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 1. Start the first job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_hardl1ace_dice_ce_10bin --seed 12345 --cpus "8-15" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 2. Start the second job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_hardl1ace_dice_ce_50bin --seed 12345 --cpus "8-15" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 3. Start the third job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_hardl1ace_dice_ce_100bin --seed 12345 --cpus "8-15" --gpu 0
# docker wait "$(docker ps --latest --quiet)"
