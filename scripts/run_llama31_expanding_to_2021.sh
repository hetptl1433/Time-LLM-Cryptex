#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXPERIMENT="era_sensitivity_2022_2025_llama31"
TEST_FILE="dataset/splits/test_2022_2025.csv"
TEST_START="2022-01-01"
TEST_END="2025-12-31"

mkdir -p logs outputs results

# start_year:train_start_date
ERAS=(
  "2021:2021-01-01"
  "2020:2020-01-01"
  "2019:2019-01-01"
  "2018:2018-01-01"
  "2017:2017-01-01"
  "2016:2016-01-01"
  "2015:2015-01-01"
  "2014:2014-01-01"
  "2013:2013-10-06"
)

is_logged() {
  local run_name="$1"
  python3 - "$run_name" <<'PY'
import sys
from pathlib import Path
import pandas as pd

run_name = sys.argv[1]
p = Path("results/experiment_tracker_llama31.csv")
if not p.exists():
    print("no")
    raise SystemExit(0)
try:
    df = pd.read_csv(p)
except Exception:
    print("no")
    raise SystemExit(0)
if "run_name" not in df.columns:
    print("no")
else:
    print("yes" if (df["run_name"].astype(str) == run_name).any() else "no")
PY
}

for item in "${ERAS[@]}"; do
  IFS=':' read -r START_YEAR TRAIN_START <<<"$item"
  RUN_NAME="era_${START_YEAR}_2021_llama31"
  TRAIN_FILE="dataset/splits/train_${START_YEAR}_2021.csv"
  OUT_DIR="outputs/${RUN_NAME}_test_2022_2025"
  LOG_FILE="logs/${RUN_NAME}.log"

  if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "[$(date -u '+%F %T')] missing split: $TRAIN_FILE, skipping"
    continue
  fi

  if [[ "$(is_logged "$RUN_NAME")" == "yes" ]]; then
    echo "[$(date -u '+%F %T')] already logged: $RUN_NAME, skipping" | tee -a "$LOG_FILE"
    continue
  fi

  echo "[$(date -u '+%F %T')] start training: $RUN_NAME" | tee -a "$LOG_FILE"
  if ! python3 run_main.py \
    --model_id "$RUN_NAME" \
    --data CRYPTEX \
    --root_path ./dataset \
    --data_path "splits/train_${START_YEAR}_2021.csv" \
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

  echo "[$(date -u '+%F %T')] log tracker: $RUN_NAME" | tee -a "$LOG_FILE"
  if ! python3 utils/experiment_tracker.py \
    --experiment_name "$EXPERIMENT" \
    --run_name "$RUN_NAME" \
    --inference_csv "$OUT_DIR/inference.csv" \
    --train_start "$TRAIN_START" \
    --train_end "2021-12-31" \
    --test_start "$TEST_START" \
    --test_end "$TEST_END" \
    --notes "llama31, train=${START_YEAR}-2021, future_test=2022-2025" \
    --output_csv results/experiment_tracker_llama31.csv \
    --output_xlsx results/experiment_tracker_llama31.xlsx \
    --log_inference_metrics_to_mlflow >>"$LOG_FILE" 2>&1; then
    echo "[$(date -u '+%F %T')] tracker update failed: $RUN_NAME" | tee -a "$LOG_FILE"
    continue
  fi

  echo "[$(date -u '+%F %T')] completed: $RUN_NAME" | tee -a "$LOG_FILE"
done

echo "[$(date -u '+%F %T')] llama31 expanding-window batch complete"

python3 utils/find_optimal_growing_window_llama31.py || true
