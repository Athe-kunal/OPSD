#!/bin/bash
# Runs all five OPSD token-selection experiments sequentially.
# Results are logged to WandB project "OPSD" with distinct run names per experiment.
# Usage: bash scripts/run_all_experiments.sh
# Leave running overnight; each experiment picks up after the previous one finishes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(
    "run_opsd_1b.sh"
    "run_opsd_1b_first_sentence.sh"
    "run_opsd_1b_middle_sentences.sh"
    "run_opsd_1b_last_sentence.sh"
    "run_opsd_1b_paragraph_first_token.sh"
)

TOTAL=${#EXPERIMENTS[@]}
FAILED=()

echo "========================================"
echo " OPSD experiment suite — $TOTAL runs"
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
    if bash "$SCRIPT_DIR/$SCRIPT" 2>&1 | tee "$LOG_FILE"; then
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
