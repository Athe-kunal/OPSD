#!/bin/bash
# Runs all 9 OPSD mixed-beta positional experiments sequentially:
#   1 baseline + 8 combinations of (beta_first, beta_middle, beta_last) in {0,1}^3
#
# beta=0 → forward KL  (KL(teacher||student), good when student is overconfident)
# beta=1 → reverse KL  (KL(student||teacher), good when teacher leads)
#
# Analysis finding: last > middle >> first for teacher signal strength.
# Motivated combination: first→0, middle→1, last→1
#
# Usage:
#   bash scripts_positions/run_all_mixed_expts.sh [per_device_batch] [grad_accum] [max_steps] [save_steps] [train_gpu] [rollout_gpu]
# Example:
#   bash scripts_positions/run_all_mixed_expts.sh 8 2 100 25 2 3

set -euo pipefail

PER_DEVICE_BATCH=${1:-4}
GRAD_ACCUM=${2:-4}
MAX_STEPS=${3:-100}
SAVE_STEPS=${4:-25}
TRAIN_GPU=${5:-2}
ROLLOUT_GPU=${6:-3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_DIR/logs/mixed_expts"
mkdir -p "$LOG_DIR"


FAILED=()
RUN_NUM=0
TOTAL=8

beta_to_kl() { [ "$1" = "0" ] && echo "fwd" || echo "rev"; }

run_experiment() {
    local bf="$1"
    local bm="$2"
    local bl="$3"
    local kl_f=$(beta_to_kl "$bf")
    local kl_m=$(beta_to_kl "$bm")
    local kl_l=$(beta_to_kl "$bl")
    RUN_NUM=$((RUN_NUM + 1))

    echo ""
    echo "----------------------------------------"
    echo "[$RUN_NUM/$TOTAL] first=${kl_f} middle=${kl_m} last=${kl_l}"
    echo "  Time: $(date)"
    echo "----------------------------------------"

    if bash "$SCRIPT_DIR/run_fwd_rev_rev.sh" \
        "$kl_f" "$kl_m" "$kl_l" \
        "$PER_DEVICE_BATCH" "$GRAD_ACCUM" "$MAX_STEPS" "$SAVE_STEPS" \
        "$TRAIN_GPU" "$ROLLOUT_GPU" "$TRAIN_GPU" "$ROLLOUT_GPU"; then
        echo "[$RUN_NUM/$TOTAL] DONE  ($(date))"
    else
        echo "[$RUN_NUM/$TOTAL] FAILED  ($(date))"
        FAILED+=("${kl_f}_${kl_m}_${kl_l}")
    fi
}

echo "========================================"
echo " OPSD mixed-beta positional ablations — $TOTAL runs"
echo " Iterating over (beta_first, beta_middle, beta_last) in {0,1}^3"
echo " per_device_batch=$PER_DEVICE_BATCH  grad_accum=$GRAD_ACCUM  max_steps=$MAX_STEPS"
echo " train_gpu=$TRAIN_GPU  rollout_gpu=$ROLLOUT_GPU"
echo " Logs: $LOG_DIR"
echo " Start: $(date)"
echo "========================================"

# All 8 combinations of (beta_first, beta_middle, beta_last) in {0,1}^3
for BF in 0 1; do
    for BM in 0 1; do
        for BL in 0 1; do
            run_experiment "$BF" "$BM" "$BL"
        done
    done
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
