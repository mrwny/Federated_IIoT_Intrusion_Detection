#!/bin/bash

# Runs the Flower simulation across the range of client counts for both IID and
# non-IID partitioning, across all model types and datasets, on the 3 seeds.
# Results are saved under results/Federated/{dataset}
#
# Usage:
#   ./run_client_sweep.sh

set -euo pipefail

# Simulation Configuration
CLIENT_COUNTS=(2 4 8 12 16 20)
FRACTIONS=(1.0 0.4)
PARTITION_TYPES=("iid" "non-iid")
SEEDS=(79397 62529 70814)
FEDERATION="local-simulation-gpu"
APP_DIR="."

MODEL_TYPES=("DNN" "CNN" "XGBoost" "RandomForest" "AttentionXGBoost")
DATASETS=("DataSense" "Edge-IIoT")

NUM_ROUNDS=20
LOCAL_EPOCHS=5
LR=0.001

get_strategies_for_model() {
    case "$1" in
        DNN|CNN)           echo "FedAvg FedProx" ;;
        XGBoost)           echo "FedXgbBagging"  ;;
        AttentionXGBoost)  echo "AttentionFedXgb" ;;
        RandomForest)      echo "FedRF"          ;;
    esac
}

get_num_rounds_for_model() {
    case "$1" in
        RandomForest) echo 1  ;;
        *)            echo "${NUM_ROUNDS}" ;;
    esac
}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="sweep_logs/${TIMESTAMP}"
SUMMARY_FILE="${LOG_DIR}/sweep_summary.csv"

DRY_RUN=false

mkdir -p "${LOG_DIR}"

echo "dataset,model,partition_type,num_clients,fraction,strategy,num_rounds,seed,status,duration_sec,log_file" > "${SUMMARY_FILE}"

TOTAL_RUNS=0
for MODEL in "${MODEL_TYPES[@]}"; do
    STRATEGIES_STR=$(get_strategies_for_model "${MODEL}")
    for _ in ${STRATEGIES_STR}; do
        for P_TYPE in "${PARTITION_TYPES[@]}"; do
            for _ in "${CLIENT_COUNTS[@]}"; do
                for FRAC in "${FRACTIONS[@]}"; do
                    # Skip counting partial fractions for IID data
                    if [ "${P_TYPE}" = "iid" ] && [ "${FRAC}" != "1.0" ]; then
                        continue
                    fi
                    for _ in "${SEEDS[@]}"; do
                        for _ in "${DATASETS[@]}"; do
                            TOTAL_RUNS=$((TOTAL_RUNS + 1))
                        done
                    done
                done
            done
        done
    done
done

CURRENT_RUN=0
FAILED=0
SUCCEEDED=0

echo "============================================================================"
echo "  Federated NIDS - Client Scalability Sweep (Multi-Seed)"
echo "============================================================================"
echo "  Client counts  : ${CLIENT_COUNTS[*]}"
echo "  Fractions      : ${FRACTIONS[*]}"
echo "  Partition types : ${PARTITION_TYPES[*]}"
echo "  Model types     : ${MODEL_TYPES[*]}"
echo "  Datasets        : ${DATASETS[*]}"
echo "  Seeds           : ${SEEDS[*]}"
echo "  Rounds (default): ${NUM_ROUNDS}  (RF uses 1)"
echo "  Local epochs    : ${LOCAL_EPOCHS}"
echo "  Total runs      : ${TOTAL_RUNS}"
echo "  Log directory   : ${LOG_DIR}"
echo "============================================================================"
echo ""

for DATASET in "${DATASETS[@]}"; do
for MODEL in "${MODEL_TYPES[@]}"; do
  STRATEGIES_STR=$(get_strategies_for_model "${MODEL}")
  ROUNDS=$(get_num_rounds_for_model "${MODEL}")

  for STRATEGY in ${STRATEGIES_STR}; do
    for PARTITION in "${PARTITION_TYPES[@]}"; do
      for N_CLIENTS in "${CLIENT_COUNTS[@]}"; do
        for FRACTION in "${FRACTIONS[@]}"; do
        
          # Skip running partial fractions for IID data (to save vast amounts of time)
          # IID with partial participation isn't interesting for performance comparisons.
          if [ "${PARTITION}" = "iid" ] && [ "${FRACTION}" != "1.0" ]; then
              continue
          fi

          for SEED in "${SEEDS[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            RUN_LABEL="${DATASET}_${MODEL}_${STRATEGY}_${PARTITION}_${N_CLIENTS}clients_frac${FRACTION}_seed${SEED}"
            LOG_FILE="${LOG_DIR}/${RUN_LABEL}.log"

            echo "--------------------------------------------------------------"
            echo "[${CURRENT_RUN}/${TOTAL_RUNS}] dataset=${DATASET}  model=${MODEL}  strategy=${STRATEGY}  partition=${PARTITION}  clients=${N_CLIENTS}  frac=${FRACTION}  seed=${SEED}  rounds=${ROUNDS}"
            echo "--------------------------------------------------------------"

            CMD="flwr run ${APP_DIR} ${FEDERATION} \\
                --run-config 'partition-type=\"${PARTITION}\" strategy=\"${STRATEGY}\" model-type=\"${MODEL}\" dataset=\"${DATASET}\" num-server-rounds=${ROUNDS} local-epochs=${LOCAL_EPOCHS} lr=${LR} proximal-mu=1.0 fraction-train=${FRACTION} seed=${SEED}' \\
                --federation-config 'options.num-supernodes=${N_CLIENTS}' \\
                --stream"

            if $DRY_RUN; then
                echo "  [DRY RUN] Would execute:"
                echo "    ${CMD}"
                echo ""
                echo "${DATASET},${MODEL},${PARTITION},${N_CLIENTS},${FRACTION},${STRATEGY},${ROUNDS},${SEED},dry-run,0,${LOG_FILE}" >> "${SUMMARY_FILE}"
                continue
            fi

            # resuming logic
            PROJECT_ROOT="/cs/student/projects3/2023/myassini/iiot_nids"
            if [ "${STRATEGY}" = "FedProx" ]; then
                EXPECTED_OUT="${PROJECT_ROOT}/results/Federated/${DATASET}/${STRATEGY}/${MODEL}/${PARTITION}/${N_CLIENTS}/frac_${FRACTION}/mu_1.0/seed_${SEED}/results.txt"
            elif [ "${STRATEGY}" = "AttentionFedXgb" ]; then
                EXPECTED_OUT="${PROJECT_ROOT}/results/Federated/${DATASET}/${STRATEGY}/XGBoost/${PARTITION}/${N_CLIENTS}/frac_${FRACTION}/seed_${SEED}/results.txt"
            else
                EXPECTED_OUT="${PROJECT_ROOT}/results/Federated/${DATASET}/${STRATEGY}/${MODEL}/${PARTITION}/${N_CLIENTS}/frac_${FRACTION}/seed_${SEED}/results.txt"
            fi

            if [ -f "${EXPECTED_OUT}" ]; then
                echo "  Run already completed (found results.txt). Skipping..."
                echo "${DATASET},${MODEL},${PARTITION},${N_CLIENTS},${FRACTION},${STRATEGY},${ROUNDS},${SEED},skipped_existing,0,${EXPECTED_OUT}" >> "${SUMMARY_FILE}"
                SUCCEEDED=$((SUCCEEDED + 1))
                echo ""
                continue
            fi
            # --------------------

            echo "  Starting at $(date '+%H:%M:%S')..."
            START_TIME=$(date +%s)

            set +e
            flwr run ${APP_DIR} ${FEDERATION} \
                --run-config "partition-type=\"${PARTITION}\" strategy=\"${STRATEGY}\" model-type=\"${MODEL}\" dataset=\"${DATASET}\" num-server-rounds=${ROUNDS} local-epochs=${LOCAL_EPOCHS} lr=${LR} proximal-mu=1.0 fraction-train=${FRACTION} seed=${SEED}" \
                --federation-config "options.num-supernodes=${N_CLIENTS}" \
                --stream \
                2>&1 | tee "${LOG_FILE}"
            EXIT_CODE=${PIPESTATUS[0]}
            set -e

            ray stop --force > /dev/null 2>&1 || true
            pkill -f "flower" || true
            sleep 10 # Give the OS time to reclaim the RAM, too many OOM errors...

            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))

            if [ ${EXIT_CODE} -eq 0 ]; then
                STATUS="success"
                SUCCEEDED=$((SUCCEEDED + 1))
                echo "  Completed in ${DURATION}s"
            else
                STATUS="failed"
                FAILED=$((FAILED + 1))
                echo "  FAILED (exit code ${EXIT_CODE}) after ${DURATION}s -- see ${LOG_FILE}"
            fi

            echo "${DATASET},${MODEL},${PARTITION},${N_CLIENTS},${FRACTION},${STRATEGY},${ROUNDS},${SEED},${STATUS},${DURATION},${LOG_FILE}" >> "${SUMMARY_FILE}"
            echo ""
          done
        done
      done
    done
  done
done
done

echo "============================================================================"
echo "  SWEEP COMPLETE"
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
    printf "%-12s  %-14s  %-12s  %-8s  %-8s  %-14s  %-6s  %-6s  %-10s  %-10s\n" "Dataset" "Model" "Partition" "Clients" "Fraction" "Strategy" "Rounds" "Seed" "Status" "Duration(s)"
    printf "%-12s  %-14s  %-12s  %-8s  %-8s  %-14s  %-6s  %-6s  %-10s  %-10s\n" "-----------" "-------------" "----------" "-------" "--------" "-------------" "------" "-----" "--------" "-----------"
    tail -n +2 "${SUMMARY_FILE}" | while IFS=',' read -r ds model pt nc frac strat rounds seed status dur logf; do
        printf "%-12s  %-14s  %-12s  %-8s  %-8s  %-14s  %-6s  %-6s  %-10s  %-10s\n" "${ds}" "${model}" "${pt}" "${nc}" "${frac}" "${strat}" "${rounds}" "${seed}" "${status}" "${dur}"
    done
fi

if ! $DRY_RUN; then
    echo ""
    echo "Aggregating federated results across seeds..."
    python aggregate_federated_results.py || echo "Aggregation failed (results may not be ready yet)"
fi
