# Finds the optimal proximal_mu for FedProx by testing a range of values
# with a fixed number of clients on non-IID data. Runs are faster than the
# full client sweep since only one variable changes.
#
# Usage:
#   ./run_mu_sweep.sh              # Run full sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MU_VALUES=(0.001 0.01 0.05 0.1 0.5 1.0)
SEEDS=(79397 62529 70814)
DATASETS=("DataSense" "Edge-IIoT")
NUM_CLIENTS=20                  # Fixed - pick a high-stress client count
PARTITION_TYPE="non-iid"        # Non-IID is where mu matters
FEDERATION="local-simulation"
APP_DIR="."

MODEL_TYPE="DNN"
NUM_ROUNDS=20
LOCAL_EPOCHS=5
LR=0.001
BATCH_SIZE=64

# 
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE -- no experiments will be executed ==="
    echo ""
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="sweep_logs/mu_sweep_${TIMESTAMP}"
SUMMARY_FILE="${LOG_DIR}/mu_sweep_summary.csv"

mkdir -p "${LOG_DIR}"

echo "dataset,proximal_mu,num_clients,partition_type,model,seed,status,duration_sec,log_file" > "${SUMMARY_FILE}"

TOTAL_RUNS=$((${#MU_VALUES[@]} * ${#SEEDS[@]} * ${#DATASETS[@]}))
CURRENT_RUN=0
FAILED=0
SUCCEEDED=0

echo "============================================================================"
echo "  Federated NIDS - Proximal Mu Sweep"
echo "============================================================================"
echo "  mu values      : ${MU_VALUES[*]}"
echo "  Seeds          : ${SEEDS[*]}"
echo "  Datasets       : ${DATASETS[*]}"
echo "  Clients        : ${NUM_CLIENTS}"
echo "  Partition      : ${PARTITION_TYPE}"
echo "  Model          : ${MODEL_TYPE}"
echo "  Rounds         : ${NUM_ROUNDS}"
echo "  Local epochs   : ${LOCAL_EPOCHS}"
echo "  Total runs     : ${TOTAL_RUNS}"
echo "  Log directory  : ${LOG_DIR}"
echo "============================================================================"
echo ""

for DATASET in "${DATASETS[@]}"; do
for MU in "${MU_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        RUN_LABEL="${DATASET}_mu_${MU}_seed_${SEED}"
        LOG_FILE="${LOG_DIR}/${RUN_LABEL}.log"

        echo "--------------------------------------------------------------"
        echo "[${CURRENT_RUN}/${TOTAL_RUNS}] dataset=${DATASET}  proximal-mu=${MU}  clients=${NUM_CLIENTS}  partition=${PARTITION_TYPE}  seed=${SEED}"
        echo "--------------------------------------------------------------"

        CMD="flwr run ${APP_DIR} ${FEDERATION} \
            --run-config 'dataset=\"${DATASET}\" partition-type=\"${PARTITION_TYPE}\" strategy=\"FedProx\" model-type=\"${MODEL_TYPE}\" num-server-rounds=${NUM_ROUNDS} local-epochs=${LOCAL_EPOCHS} lr=${LR} proximal-mu=${MU} seed=${SEED} fraction-train=0.3' \
            --federation-config 'options.num-supernodes=${NUM_CLIENTS}' \
            --stream"

        if $DRY_RUN; then
            echo "  [DRY RUN] Would execute:"
            echo "    ${CMD}"
            echo ""
            echo "${DATASET},${MU},${NUM_CLIENTS},${PARTITION_TYPE},${MODEL_TYPE},${SEED},dry-run,0,${LOG_FILE}" >> "${SUMMARY_FILE}"
            continue
        fi

        echo "  Starting at $(date '+%H:%M:%S')..."
        START_TIME=$(date +%s)

        set +e
        flwr run ${APP_DIR} ${FEDERATION} \
            --run-config "dataset=\"${DATASET}\" partition-type=\"${PARTITION_TYPE}\" strategy=\"FedProx\" model-type=\"${MODEL_TYPE}\" num-server-rounds=${NUM_ROUNDS} local-epochs=${LOCAL_EPOCHS} lr=${LR} proximal-mu=${MU} seed=${SEED} fraction-train=0.3" \
            --federation-config "options.num-supernodes=${NUM_CLIENTS}" \
            --stream \
            2>&1 | tee "${LOG_FILE}"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ ${EXIT_CODE} -eq 0 ]; then
            STATUS="success"
            SUCCEEDED=$((SUCCEEDED + 1))
            echo "  ${DATASET} mu=${MU} seed=${SEED} completed in ${DURATION}s"
        else
            STATUS="failed"
            FAILED=$((FAILED + 1))
            echo "  ${DATASET} mu=${MU} seed=${SEED} FAILED (exit code ${EXIT_CODE}) after ${DURATION}s -- see ${LOG_FILE}"
        fi

        echo "${DATASET},${MU},${NUM_CLIENTS},${PARTITION_TYPE},${MODEL_TYPE},${SEED},${STATUS},${DURATION},${LOG_FILE}" >> "${SUMMARY_FILE}"
        echo ""
    done
done
done

echo "============================================================================"
echo "  MU SWEEP COMPLETE"
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
    printf "%-12s  %-12s  %-8s  %-10s  %-10s\n" "Dataset" "mu" "Clients" "Status" "Duration(s)"
    printf "%-12s  %-12s  %-8s  %-10s  %-10s\n" "-------" "----------" "-------" "--------" "-----------"
    tail -n +2 "${SUMMARY_FILE}" | while IFS=',' read -r dataset mu nc pt model seed status dur logf; do
        printf "%-12s  %-12s  %-8s  %-10s  %-10s\n" "${dataset}" "${mu}" "${nc}" "${status}" "${dur}"
    done
    echo ""
    echo "Compare results in results/Federated/<Dataset>/FedProx/DNN/*/non-iid/${NUM_CLIENTS}/results.txt"
fi
