#!/bin/bash

BASE_MODEL="Qwen/Qwen3-1.7B"
CHECKPOINT_DIR="/home/recoverx/astarag/OPSD/data0/siyanz/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-250"
STEP=250
EXP_NAME=$(basename "$(dirname "$CHECKPOINT_DIR")")

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3 python eval/eval_math.py \
    --base_model "$BASE_MODEL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 2 \
    --wandb_project "OPSD-eval" \
    --wandb_run_name "opsd-eval-${STEP}-${EXP_NAME}-thinking"
