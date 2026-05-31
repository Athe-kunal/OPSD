#!/bin/bash

BASE_MODEL="Qwen/Qwen3-1.7B"
CHECKPOINT_DIR="data0/siyanz/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005_lastsent_topk1/checkpoint-100"
STEP=100
GPU=1
THINKING="thinking"
EXP_NAME=$(basename "$(dirname "$CHECKPOINT_DIR")")
LOG_DIR="logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

THINKING_FLAG=""
THINKING_LABEL="thinking"
if [ "$THINKING" = "nonthinking" ]; then
    THINKING_FLAG="--no_thinking"
    THINKING_LABEL="nonthinking"
fi

RUN_NAME="opsd-eval-step${STEP}-${EXP_NAME}-${THINKING_LABEL}"
LOG_FILE="${LOG_DIR}/step${STEP}-${THINKING_LABEL}.log"

echo "[$(date '+%H:%M:%S')] Starting step=${STEP} (${THINKING_LABEL}) on GPU ${GPU} → ${LOG_FILE}"

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GPU} python eval/eval_math.py \
    --base_model "$BASE_MODEL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --val_n 8 \
    --temperature 1.0 \
    --tensor_parallel_size 1 \
    --step "$STEP" \
    $THINKING_FLAG \
    --wandb_project "OPSD-eval" \
    --wandb_run_name "$RUN_NAME" \
    > "$LOG_FILE" 2>&1

echo "[$(date '+%H:%M:%S')] Done step=${STEP} (${THINKING_LABEL}) on GPU ${GPU}"
