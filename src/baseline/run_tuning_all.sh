#!/bin/bash

# This script sequentially runs the Bayesian hyperparameter tuning for
# tree-models on both datasets.
#
# Usage:
#   ./run_tuning_all.sh

set -euo pipefail

# Ensure we are running from the script's directory
cd "$(dirname "$0")"

echo "==============================================="
echo "  Starting Hyperparameter Tuning Sweep         "
echo "==============================================="

echo "-----------------------------------------------"
echo "[1/4] Tuning XGBoost on DataSense"
echo "-----------------------------------------------"
python tuning/tune_tree_models_bayesian.py --model xgboost --dataset DataSense

echo "-----------------------------------------------"
echo "[2/4] Tuning Random Forest on DataSense"
echo "-----------------------------------------------"
python tuning/tune_tree_models_bayesian.py --model random_forest --dataset DataSense

echo "-----------------------------------------------"
echo "[3/4] Tuning XGBoost on Edge-IIoT"
echo "-----------------------------------------------"
python tuning/tune_tree_models_bayesian.py --model xgboost --dataset Edge-IIoT

echo "-----------------------------------------------"
echo "[4/4] Tuning Random Forest on Edge-IIoT"
echo "-----------------------------------------------"
python tuning/tune_tree_models_bayesian.py --model random_forest --dataset Edge-IIoT

echo "==============================================="
echo "  All 4 tuning sweeps successfully completed!  "
echo "==============================================="

