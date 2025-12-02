#!/bin/bash

# 1. Start the first job and wait for it to finish
./docker_run.sh --mode inference_eval_additional --bundle acdc17_baseline_ce_2 --seed 12345 --cpus "8-15" --gpu 0
# docker wait "$(docker ps --latest --quiet)"
./docker_run.sh --mode inference_eval_additional --bundle acdc17_softl1ace_dice_ce_2 --seed 12345 --cpus "8-15" --gpu 0
./docker_run.sh --mode temp_scale_eval_additional --bundle acdc17_softl1ace_dice_ce_2 --seed 12345 --cpus "8-15" --gpu 0

./docker_run.sh --mode inference_eval_additional --bundle amos22_baseline_ce_nl --seed 12345 --cpus "8-15" --gpu 0

./docker_run.sh --mode inference_eval_additional --bundle amos22_baseline_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0
./docker_run.sh --mode temp_scale_eval_additional --bundle amos22_baseline_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0

./docker_run.sh --mode inference_eval_additional --bundle amos22_hardl1ace_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0
./docker_run.sh --mode temp_scale_eval_additional --bundle amos22_hardl1ace_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0

./docker_run.sh --mode inference_eval_additional --bundle amos22_softl1ace_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0
./docker_run.sh --mode temp_scale_eval_additional --bundle amos22_softl1ace_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0

./docker_run.sh --mode inference_eval_additional --bundle brats21_baseline_ce_nl --seed 12345 --cpus "8-15" --gpu 0

./docker_run.sh --mode inference_eval_additional --bundle brats21_baseline_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0
./docker_run.sh --mode temp_scale_eval_additional --bundle brats21_baseline_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0

./docker_run.sh --mode inference_eval_additional --bundle brats21_hardl1ace_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0
./docker_run.sh --mode temp_scale_eval_additional --bundle brats21_hardl1ace_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0

./docker_run.sh --mode inference_eval_additional --bundle brats21_softl1ace_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0
./docker_run.sh --mode temp_scale_eval_additional --bundle brats21_softl1ace_dice_ce_nl --seed 12345 --cpus "8-15" --gpu 0