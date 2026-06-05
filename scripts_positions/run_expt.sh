#!/bin/bash
# Run a single OPSD positional mixed-beta experiment.
# Accepts KL direction per third as "fwd" or "rev", maps to beta 0 or 1.
# After training, evaluates all saved checkpoints in parallel (2 at a time).
#
# Usage:
#   bash scripts_positions/run_fwd_rev_rev.sh <kl_first> <kl_middle> <kl_last> [per_device_batch] [grad_accum] [max_steps] [save_steps] [train_gpu] [rollout_gpu] [eval_gpu0] [eval_gpu1]
#
#   kl_first / kl_middle / kl_last: "fwd" or "rev"
#
# Examples:
#   bash scripts_positions/run_fwd_rev_rev.sh fwd rev rev 8 2 100 25 2 3 2 3
#   bash scripts_positions/run_fwd_rev_rev.sh rev rev rev 8 2 100 25 2 3 2 3

set -euo pipefail

# ── KL direction args ─────────────────────────────────────────────────────────

# fwd: 0
# rev: 1

# Done
# 011
# 100
# 111

# Running
# 110

# Remaining
# 010
# 001
# 010
# 101
# 101

KL_FIRST=${1:-rev}
KL_MIDDLE=${2:-rev}
KL_LAST=${3:-fwd}

kl_to_beta() {
    case "$1" in
        fwd) echo 0 ;;
        rev) echo 1 ;;
        *)
            echo "ERROR: KL direction must be 'fwd' or 'rev', got '$1'" >&2
            exit 1
            ;;
    esac
}

BETA_FIRST=$(kl_to_beta "$KL_FIRST")
BETA_MIDDLE=$(kl_to_beta "$KL_MIDDLE")
BETA_LAST=$(kl_to_beta "$KL_LAST")

# ── Other args ────────────────────────────────────────────────────────────────

PER_DEVICE_BATCH=${4:-4}
GRAD_ACCUM=${5:-4}
MAX_STEPS=${6:-100}
SAVE_STEPS=${7:-25}
TRAIN_GPU=${8:-0}
ROLLOUT_GPU=${9:-1}
EVAL_GPU0=${10:-0}
EVAL_GPU1=${11:-1}

RUN_CONFIG="qwen31b_pos_bf${BETA_FIRST}_bm${BETA_MIDDLE}_bl${BETA_LAST}"
BASE_MODEL="Qwen/Qwen3-1.7B"
OUTPUT_DIR="data0/siyanz/opsd"
EXP_DIR="${OUTPUT_DIR}/${RUN_CONFIG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_DIR/logs/${RUN_CONFIG}"
mkdir -p "$LOG_DIR"

# ── Training ──────────────────────────────────────────────────────────────────

echo "========================================"
echo " TRAINING: $RUN_CONFIG"
echo " first=${KL_FIRST}(beta=${BETA_FIRST})  middle=${KL_MIDDLE}(beta=${BETA_MIDDLE})  last=${KL_LAST}(beta=${BETA_LAST})"
echo " max_steps=$MAX_STEPS  save_steps=$SAVE_STEPS"
echo " train_gpu=$TRAIN_GPU  rollout_gpu=$ROLLOUT_GPU"
echo " Start: $(date)"
echo "========================================"

CUDA_VISIBLE_DEVICES=$TRAIN_GPU,$ROLLOUT_GPU accelerate launch \
    --config_file "$REPO_DIR/accelerate.yaml" \
    --num_processes 2 \
    --main_process_port 12949 \
    "$REPO_DIR/opsd_train.py" \
    --model_name_or_path "$BASE_MODEL" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 1 \
    --max_completion_length 1024 \
    --save_steps "$SAVE_STEPS" \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta "$BETA_FIRST" \
    --beta_first "$BETA_FIRST" \
    --beta_middle "$BETA_MIDDLE" \
    --beta_last "$BETA_LAST" \
    --token_selection_mode baseline \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_tensor_parallel_size 2 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD-position \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""
echo "Training done. Waiting 30s for GPU memory to fully release before eval..."
sleep 60

echo "Starting checkpoint evaluation..."

# ── Evaluation ────────────────────────────────────────────────────────────────

run_checkpoint() {
    local STEP=$1
    local GPU=$2
    local THINKING=$3   # "thinking" or "nonthinking"
    local CHECKPOINT_DIR="${EXP_DIR}/checkpoint-${STEP}"

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "[$(date '+%H:%M:%S')] Skipping step=${STEP} — checkpoint dir not found: $CHECKPOINT_DIR"
        return
    fi

    local THINKING_FLAG=""
    local THINKING_LABEL="thinking"
    if [ "$THINKING" = "nonthinking" ]; then
        THINKING_FLAG="--no_thinking"
        THINKING_LABEL="nonthinking"
    fi

    local RUN_NAME="opsd-eval-step${STEP}-${RUN_CONFIG}-${THINKING_LABEL}"
    local LOG_FILE="${LOG_DIR}/eval_step${STEP}_${THINKING_LABEL}.log"

    echo "[$(date '+%H:%M:%S')] Eval step=${STEP} (${THINKING_LABEL}) on GPU ${GPU} → ${LOG_FILE}"

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GPU} python "$REPO_DIR/eval/eval_math.py" \
        --base_model "$BASE_MODEL" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --val_n 8 \
        --temperature 1.0 \
        --tensor_parallel_size 1 \
        --step "$STEP" \
        $THINKING_FLAG \
        --wandb_project "OPSD-position-eval" \
        --wandb_run_name "$RUN_NAME" \
        > "$LOG_FILE" 2>&1

    echo "[$(date '+%H:%M:%S')] Done step=${STEP} (${THINKING_LABEL}) on GPU ${GPU}"
}

# Collect all checkpoint steps that were actually saved
STEPS=()
for dir in "$EXP_DIR"/checkpoint-*/; do
    step=$(basename "$dir" | sed 's/checkpoint-//')
    STEPS+=("$step")
done

# Sort numerically
IFS=$'\n' STEPS=($(sort -n <<<"${STEPS[*]}")); unset IFS

if [ ${#STEPS[@]} -eq 0 ]; then
    echo "No checkpoints found in $EXP_DIR — skipping evaluation."
    exit 0
fi

echo ""
echo "========================================"
echo " EVALUATION: ${#STEPS[@]} checkpoints found"
echo " Steps: ${STEPS[*]}"
echo " eval_gpu0=$EVAL_GPU0  eval_gpu1=$EVAL_GPU1"
echo " Mode: thinking"
echo "========================================"

# Run 2 checkpoints at a time, alternating between the two eval GPUs
i=0
while [ $i -lt ${#STEPS[@]} ]; do
    STEP0="${STEPS[$i]}"
    STEP1="${STEPS[$i+1]:-}"

    if [ -n "$STEP1" ]; then
        echo "--- Batch: checkpoint-${STEP0} (GPU $EVAL_GPU0) and checkpoint-${STEP1} (GPU $EVAL_GPU1) ---"
        run_checkpoint "$STEP0" "$EVAL_GPU0" "thinking" &
        run_checkpoint "$STEP1" "$EVAL_GPU1" "thinking" &
        wait
    else
        echo "--- Batch: checkpoint-${STEP0} (GPU $EVAL_GPU0) ---"
        run_checkpoint "$STEP0" "$EVAL_GPU0" "thinking"
    fi

    i=$((i + 2))
done

echo ""
echo "========================================"
echo " All done: $(date)"
echo " Logs: $LOG_DIR"
echo "========================================"
