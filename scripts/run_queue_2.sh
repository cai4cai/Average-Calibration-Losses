#!/bin/bash

# 1. Start the first job and wait for it to finish
# ./docker_run_new.sh --mode train --bundle acdc17_baseline_dice_ce_2 --seed 12345 --cpus "16-23" --gpu 1
# docker wait "$(docker ps --latest --quiet)"


# ./docker_run_new.sh --mode inference_eval_additional --bundle kits23_baseline_dice_ce_nl --seed 12345 --cpus "16-23" --gpu 1
# ./docker_run_new.sh --mode temp_scale_eval_additional --bundle kits23_baseline_dice_ce_nl --seed 12345 --cpus "16-23" --gpu 1

# ./docker_run_new.sh --mode inference_eval_additional --bundle kits23_hardl1ace_dice_ce_nl --seed 12345 --cpus "16-23" --gpu 1
# ./docker_run_new.sh --mode temp_scale_eval_additional --bundle kits23_hardl1ace_dice_ce_nl --seed 12345 --cpus "16-23" --gpu 1

# ./docker_run_new.sh --mode inference_eval_additional --bundle kits23_softl1ace_dice_ce_nl --seed 12345 --cpus "16-23" --gpu 1
# ./docker_run_new.sh --mode temp_scale_eval_additional --bundle kits23_softl1ace_dice_ce_nl --seed 12345 --cpus "16-23" --gpu 1

./docker_run_new.sh --mode temp_scale_eval_additional --bundle kits23_baseline_ce_nl --seed 12345 --cpus "16-23" --gpu 1