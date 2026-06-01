#!/bin/bash

BASE_MODEL="Qwen/Qwen3-1.7B"
EXP_DIR="data0/siyanz/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005_firstsent_topk2"
EXP_NAME=$(basename "$EXP_DIR")
LOG_DIR="logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

run_checkpoint() {
    local STEP=$1
    local GPU=$2
    local THINKING=$3  # "thinking" or "nonthinking"
    local CHECKPOINT_DIR="${EXP_DIR}/checkpoint-${STEP}"

    local THINKING_FLAG=""
    local THINKING_LABEL="thinking"
    if [ "$THINKING" = "nonthinking" ]; then
        THINKING_FLAG="--no_thinking"
        THINKING_LABEL="nonthinking"
    fi

    local RUN_NAME="opsd-eval-step${STEP}-${EXP_NAME}-${THINKING_LABEL}"
    local LOG_FILE="${LOG_DIR}/step${STEP}-${THINKING_LABEL}.log"

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
}

# Run thinking mode evals: 2 checkpoints at a time, one per GPU
echo "=== Thinking mode ==="

echo "--- Batch 1: checkpoint-25 (GPU 2) and checkpoint-50 (GPU 3) ---"
run_checkpoint 25 0 "thinking" &
run_checkpoint 50 1 "thinking" &
wait

echo "--- Batch 2: checkpoint-75 (GPU 2) and checkpoint-100 (GPU 3) ---"
run_checkpoint 75 0 "thinking" &
run_checkpoint 100 1 "thinking" &
wait

echo "=== All checkpoints evaluated. Logs in ${LOG_DIR} ==="
