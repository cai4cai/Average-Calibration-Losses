#!/bin/bash

# 1. Start the first job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_hardl1ace_dice_ce_dsc1_ce1_ace2 --seed 12345 --cpus "16-23" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 2. Start the second job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_hardl1ace_dice_ce_dsc2_ce2_ace1 --seed 12345 --cpus "16-23" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 3. Start the third job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_softl1ace_dice_ce_10bin --seed 12345 --cpus "16-23" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 1. Start the first job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_softl1ace_dice_ce_50bin --seed 12345 --cpus "16-23" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 2. Start the second job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_softl1ace_dice_ce_100bin --seed 12345 --cpus "16-23" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 3. Start the third job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_softl1ace_dice_ce_dsc1_ce1_ace2 --seed 12345 --cpus "16-23" --gpu 0
# docker wait "$(docker ps --latest --quiet)"

# 3. Start the third job and wait for it to finish
./docker_run.sh --mode inference_eval --bundle acdc17_softl1ace_dice_ce_dsc2_ce2_ace1 --seed 12345 --cpus "16-23" --gpu 0
# docker wait "$(docker ps --latest --quiet)"