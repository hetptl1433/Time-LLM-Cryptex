#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

EXPERIMENT="era_sensitivity_2022_2025_llama31"
TEST_FILE="dataset/splits/test_2022_2025.csv"
TRACKER_CSV="results/experiment_tracker_llama31.csv"
TRACKER_XLSX="results/experiment_tracker_llama31.xlsx"

mkdir -p logs outputs results

is_logged() {
  local run_name="$1"
  python3 - "$run_name" "$TRACKER_CSV" <<'PY'
import sys
from pathlib import Path
import pandas as pd
run_name=sys.argv[1]
tracker=Path(sys.argv[2])
if not tracker.exists():
    print('no')
    raise SystemExit(0)
try:
    df=pd.read_csv(tracker)
except Exception:
    print('no')
    raise SystemExit(0)
if 'run_name' not in df.columns:
    print('no')
else:
    print('yes' if (df['run_name'].astype(str)==run_name).any() else 'no')
PY
}

for START in 2013 2014 2015 2016; do
  TRAIN_START="${START}-01-01"
  if [[ "$START" == "2013" ]]; then
    TRAIN_START="2013-10-06"
  fi

  RUN_NAME="era_${START}_2021_llama31"
  TRAIN_FILE="dataset/splits/train_${START}_2021.csv"
  OUT_DIR="outputs/${RUN_NAME}_test_2022_2025"
  INFER_CSV="${OUT_DIR}/inference.csv"
  LOG_FILE="logs/${RUN_NAME}.log"

  if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "[$(date -u '+%F %T')] missing split: $TRAIN_FILE, skipping" | tee -a "$LOG_FILE"
    continue
  fi

  if [[ "$(is_logged "$RUN_NAME")" == "yes" ]]; then
    echo "[$(date -u '+%F %T')] already logged: $RUN_NAME, skipping" | tee -a "$LOG_FILE"
    continue
  fi

  if [[ ! -f "$INFER_CSV" ]]; then
    echo "[$(date -u '+%F %T')] start training: $RUN_NAME" | tee -a "$LOG_FILE"
    if ! python3 run_main.py \
      --model_id "$RUN_NAME" \
      --data CRYPTEX \
      --root_path ./dataset \
      --data_path "splits/train_${START}_2021.csv" \
      --train_epochs 3 \
      --batch_size 2 \
      --seq_len 32 \
      --pred_len 8 \
      --llm_model LLAMA3.1 \
      --llm_layers 2 \
      --enc_in 5 \
      --num_workers 0 \
      --experiment_name "$EXPERIMENT" >>"$LOG_FILE" 2>&1; then
      echo "[$(date -u '+%F %T')] training failed: $RUN_NAME" | tee -a "$LOG_FILE"
      continue
    fi

    echo "[$(date -u '+%F %T')] start inference: $RUN_NAME" | tee -a "$LOG_FILE"
    if ! python3 run_inference.py \
      --model_id "$RUN_NAME" \
      --llm_model LLAMA3.1 \
      --experiment_name "$EXPERIMENT" \
      --mlflow_tracking_uri sqlite:///mlflow.db \
      --data_path "./$TEST_FILE" \
      --save_path "./$OUT_DIR" >>"$LOG_FILE" 2>&1; then
      echo "[$(date -u '+%F %T')] inference failed: $RUN_NAME" | tee -a "$LOG_FILE"
      continue
    fi
  else
    echo "[$(date -u '+%F %T')] inference csv exists, skip train/infer: $RUN_NAME" | tee -a "$LOG_FILE"
  fi

  echo "[$(date -u '+%F %T')] update tracker: $RUN_NAME" | tee -a "$LOG_FILE"
  if ! python3 utils/experiment_tracker.py \
    --experiment_name "$EXPERIMENT" \
    --run_name "$RUN_NAME" \
    --inference_csv "$INFER_CSV" \
    --train_start "$TRAIN_START" \
    --train_end "2021-12-31" \
    --test_start "2022-01-01" \
    --test_end "2025-12-31" \
    --notes "llama31, growing window ${START}-2021, test=2022-2025" \
    --output_csv "$TRACKER_CSV" \
    --output_xlsx "$TRACKER_XLSX" \
    --log_inference_metrics_to_mlflow >>"$LOG_FILE" 2>&1; then
    echo "[$(date -u '+%F %T')] tracker update failed: $RUN_NAME" | tee -a "$LOG_FILE"
    continue
  fi

  echo "[$(date -u '+%F %T')] completed: $RUN_NAME" | tee -a "$LOG_FILE"
done

echo "[$(date -u '+%F %T')] backfill batch done"
