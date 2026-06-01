#!/bin/bash
# Runs all five OPSD token-selection experiments sequentially.
# Results are logged to WandB project "OPSD" with distinct run names per experiment.
# Usage: bash scripts/run_all_experiments.sh [per_device_batch] [grad_accum] [max_steps] [save_steps] [train_gpu] [rollout_gpu]
# Example (GPUs 0 and 1): bash scripts/run_all_experiments.sh 4 4 250 250 0 1
# Leave running overnight; each experiment picks up after the previous one finishes.

set -euo pipefail

PER_DEVICE_BATCH=${1:-4}
GRAD_ACCUM=${2:-4}
MAX_STEPS=${3:-100}
SAVE_STEPS=${4:-25}
TRAIN_GPU=${5:-2}
ROLLOUT_GPU=${6:-3}
export TRAIN_GPU ROLLOUT_GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(
    # "run_opsd_1b.sh"
    # "run_opsd_1b_first_sentence.sh"
    # "run_opsd_1b_middle_sentences.sh"
    # "run_opsd_1b_last_sentence.sh"
    # "run_opsd_1b_paragraph_first_token.sh"
    "run_opsd_1b_first_sentence_topk2.sh"
    # "run_opsd_1b_middle_sentences_topk2.sh"
    "run_opsd_1b_last_sentence_topk2.sh"
)

TOTAL=${#EXPERIMENTS[@]}
FAILED=()

echo "========================================"
echo " OPSD experiment suite — $TOTAL runs"
echo " per_device_batch=$PER_DEVICE_BATCH  grad_accum=$GRAD_ACCUM  max_steps=$MAX_STEPS  save_steps=$SAVE_STEPS"
echo " train_gpu=$TRAIN_GPU  rollout_gpu=$ROLLOUT_GPU  (CUDA_VISIBLE_DEVICES=$TRAIN_GPU,$ROLLOUT_GPU)"
echo " Logs: $LOG_DIR"
echo " Start: $(date)"
echo "========================================"

for i in "${!EXPERIMENTS[@]}"; do
    SCRIPT="${EXPERIMENTS[$i]}"
    RUN_NUM=$((i + 1))
    LOG_FILE="$LOG_DIR/${SCRIPT%.sh}.log"

    echo ""
    echo "----------------------------------------"
    echo "[$RUN_NUM/$TOTAL] Starting: $SCRIPT"
    echo "  Log : $LOG_FILE"
    echo "  Time: $(date)"
    echo "----------------------------------------"

    # Run from repo root so relative paths (accelerate.yaml, opsd_train.py) resolve correctly.
    if bash "$SCRIPT_DIR/$SCRIPT" "$PER_DEVICE_BATCH" "$GRAD_ACCUM" "$MAX_STEPS" "$SAVE_STEPS" "$TRAIN_GPU" "$ROLLOUT_GPU" 2>&1 | tee "$LOG_FILE"; then
        echo "[$RUN_NUM/$TOTAL] DONE: $SCRIPT  ($(date))"
    else
        echo "[$RUN_NUM/$TOTAL] FAILED: $SCRIPT  ($(date))"
        FAILED+=("$SCRIPT")
        # Continue with remaining experiments even if one fails.
    fi
done

echo ""
echo "========================================"
echo " All experiments finished: $(date)"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo " Status: ALL PASSED"
else
    echo " Status: ${#FAILED[@]} FAILED"
    for f in "${FAILED[@]}"; do
        echo "   - $f"
    done
fi
echo "========================================"
