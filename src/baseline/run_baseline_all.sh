#!/bin/bash

# This script automatically runs all the baseline training and evaluations.
# Usage:
#   ./run_baseline_all.sh              # Train + evaluate all models (3 seeds)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

MODELS=("random_forest" "dnn" "cnn" "xgboost")
DATASETS=("DataSense" "Edge-IIoT")
SEEDS=(79397 62529 70814)

# Defaults from main.py
EPOCHS=30
LR=0.001
BATCH_SIZE=64

TRAIN_FLAG="--train"
DRY_RUN=false

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="baseline_logs/${TIMESTAMP}"
SUMMARY_FILE="${LOG_DIR}/baseline_summary.csv"

mkdir -p "${LOG_DIR}"

echo "partition_type,model,dataset,seed,status,duration_sec,log_file" > "${SUMMARY_FILE}"

TOTAL_RUNS=$(( ${#MODELS[@]} * ${#SEEDS[@]} * ${#DATASETS[@]} ))
CURRENT_RUN=0
FAILED=0
SUCCEEDED=0

if $DRY_RUN; then
    echo "=== DRY RUN MODE -- no experiments will be executed ==="
    echo ""
fi

echo "============================================================================"
echo "  NIDS Baseline - All Models (Multi-Seed)"
echo "============================================================================"
echo "  Models     : ${MODELS[*]}"
echo "  Datasets   : ${DATASETS[*]}"
echo "  Seeds      : ${SEEDS[*]}"
echo "  Epochs     : ${EPOCHS}"
echo "  LR         : ${LR}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Mode       : $([ -n "${TRAIN_FLAG}" ] && echo 'train + evaluate' || echo 'evaluate only')"
echo "  Total runs : ${TOTAL_RUNS}"
echo "  Log dir    : ${LOG_DIR}"
echo "============================================================================"
echo ""

for DATASET in "${DATASETS[@]}"; do
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    LOG_FILE="${LOG_DIR}/${DATASET}_${MODEL}_seed${SEED}.log"

    echo "--------------------------------------------------------------"
    echo "[${CURRENT_RUN}/${TOTAL_RUNS}] model=${MODEL}  dataset=${DATASET}  seed=${SEED}"
    echo "--------------------------------------------------------------"

    CMD="python src/baseline/main.py ${TRAIN_FLAG} --model-type ${MODEL} --dataset ${DATASET} --epochs ${EPOCHS} --lr ${LR} --batch-size ${BATCH_SIZE} --seed ${SEED}"

    if $DRY_RUN; then
        echo "  [DRY RUN] Would execute:"
        echo "    ${CMD}"
        echo ""
        echo "baseline,${MODEL},${DATASET},${SEED},dry-run,0,${LOG_FILE}" >> "${SUMMARY_FILE}"
        continue
    fi

    echo "  Starting at $(date '+%H:%M:%S')..."
    START_TIME=$(date +%s)

    set +e
    ${CMD} 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ ${EXIT_CODE} -eq 0 ]; then
        STATUS="success"
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "  ${MODEL} dataset=${DATASET} seed=${SEED} completed in ${DURATION}s"
    else
        STATUS="failed"
        FAILED=$((FAILED + 1))
        echo "  ${MODEL} dataset=${DATASET} seed=${SEED} FAILED (exit code ${EXIT_CODE}) after ${DURATION}s -- see ${LOG_FILE}"
    fi

    echo "baseline,${MODEL},${DATASET},${SEED},${STATUS},${DURATION},${LOG_FILE}" >> "${SUMMARY_FILE}"
    echo ""
  done
done
done

echo "============================================================================"
echo "  BASELINE SWEEP COMPLETE"
echo "  Succeeded: ${SUCCEEDED}/${TOTAL_RUNS}"
if [ ${FAILED} -gt 0 ]; then
    echo "  Failed:    ${FAILED}/${TOTAL_RUNS}"
fi
echo "  Summary:   ${SUMMARY_FILE}"
echo "  Logs:      ${LOG_DIR}/"
echo "============================================================================"

if ! $DRY_RUN; then
    echo ""
    echo "Results summary:"
    printf "%-16s  %-12s  %-6s  %-10s  %-10s\n" "Model" "Dataset" "Seed" "Status" "Duration(s)"
    printf "%-16s  %-12s  %-6s  %-10s  %-10s\n" "--------------" "----------" "-----" "--------" "-----------"
    tail -n +2 "${SUMMARY_FILE}" | while IFS=',' read -r pt model ds seed status dur logf; do
        printf "%-16s  %-12s  %-6s  %-10s  %-10s\n" "${model}" "${ds}" "${seed}" "${status}" "${dur}"
    done
fi

if ! $DRY_RUN; then
    echo ""
    echo "Aggregating results across seeds..."
    for DATASET in "${DATASETS[@]}"; do
        python eval/aggregate_baseline_results.py --dataset "${DATASET}" || echo "  Aggregation failed for ${DATASET} (results may not be ready yet)"
    done
fi