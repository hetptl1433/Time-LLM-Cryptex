#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <ERA_TAG> <TRAIN_START> <TRAIN_END>"
  echo "Example: $0 2018_2021 2018-01-01 2021-12-31"
  exit 1
fi

ERA_TAG="$1"
TRAIN_START="$2"
TRAIN_END="$3"

RUN_NAME="era_${ERA_TAG}_llama31"
EXPERIMENT="era_sensitivity_2022_2025_llama31"
TRAIN_FILE="dataset/splits/train_${ERA_TAG}.csv"
TEST_FILE="dataset/splits/test_2022_2025.csv"
OUT_DIR="outputs/${RUN_NAME}_test_2022_2025"
LOG_FILE="logs/${RUN_NAME}.log"

TEST_START="2022-01-01"
TEST_END="2025-12-31"

mkdir -p logs outputs results

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing train file: $TRAIN_FILE"
  exit 1
fi
if [[ ! -f "$TEST_FILE" ]]; then
  echo "Missing test file: $TEST_FILE"
  exit 1
fi

echo "[$(date -u '+%F %T')] start training: $RUN_NAME" | tee -a "$LOG_FILE"
python3 run_main.py \
  --model_id "$RUN_NAME" \
  --data CRYPTEX \
  --root_path ./dataset \
  --data_path "splits/train_${ERA_TAG}.csv" \
  --train_epochs 3 \
  --batch_size 2 \
  --seq_len 32 \
  --pred_len 8 \
  --llm_model LLAMA3.1 \
  --llm_layers 2 \
  --enc_in 5 \
  --num_workers 0 \
  --experiment_name "$EXPERIMENT" >>"$LOG_FILE" 2>&1

echo "[$(date -u '+%F %T')] start inference: $RUN_NAME" | tee -a "$LOG_FILE"
python3 run_inference.py \
  --model_id "$RUN_NAME" \
  --llm_model LLAMA3.1 \
  --experiment_name "$EXPERIMENT" \
  --mlflow_tracking_uri sqlite:///mlflow.db \
  --data_path "./$TEST_FILE" \
  --save_path "./$OUT_DIR" >>"$LOG_FILE" 2>&1

echo "[$(date -u '+%F %T')] update LLAMA31 tracker: $RUN_NAME" | tee -a "$LOG_FILE"
python3 utils/experiment_tracker.py \
  --experiment_name "$EXPERIMENT" \
  --run_name "$RUN_NAME" \
  --inference_csv "$OUT_DIR/inference.csv" \
  --train_start "$TRAIN_START" \
  --train_end "$TRAIN_END" \
  --test_start "$TEST_START" \
  --test_end "$TEST_END" \
  --notes "llama31, train=${ERA_TAG}, future_test=2022-2025" \
  --output_csv results/experiment_tracker_llama31.csv \
  --output_xlsx results/experiment_tracker_llama31.xlsx \
  --log_inference_metrics_to_mlflow >>"$LOG_FILE" 2>&1

echo "[$(date -u '+%F %T')] completed: $RUN_NAME" | tee -a "$LOG_FILE"
