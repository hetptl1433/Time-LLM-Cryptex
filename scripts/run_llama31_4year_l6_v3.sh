#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Fresh namespace so it never collides with deleted/old experiments.
EXPERIMENT="era_sensitivity_2022_2025_llama31_l6_v3"
TRACKER_CSV="results/experiment_tracker_llama31_l6_v3.csv"
TRACKER_XLSX="results/experiment_tracker_llama31_l6_v3.xlsx"

TEST_START="2022-01-01"
TEST_END="2025-12-31"
TEST_FILE="dataset/splits/test_2022_2025.csv"

# Tunables
PYTHON_BIN="${PYTHON_BIN:-python3}"     # Example override: PYTHON_BIN=/usr/bin/python3
BATCH_SIZE="${BATCH_SIZE:-8}"           # Example override: BATCH_SIZE=4
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"       # Example override: TRAIN_EPOCHS=2
MIN_FREE_GB="${MIN_FREE_GB:-15}"        # Abort a run if free disk is below this threshold

mkdir -p logs outputs results

# era_tag:train_start:train_end (fixed 4-year windows)
ERAS=(
  "2018_2021:2018-01-01:2021-12-31"
  "2017_2020:2017-01-01:2020-12-31"
  "2016_2019:2016-01-01:2019-12-31"
  "2015_2018:2015-01-01:2018-12-31"
  "2014_2017:2014-01-01:2017-12-31"
  "2013_2016:2013-10-06:2016-12-31"
)

check_free_space_gb() {
  # Returns integer free GB on filesystem containing current directory.
  df -BG . | awk 'NR==2 {gsub("G","",$4); print $4+0}'
}

is_logged() {
  local run_name="$1"
  "$PYTHON_BIN" - "$run_name" "$TRACKER_CSV" <<'PY'
import sys
from pathlib import Path
import pandas as pd

run_name = sys.argv[1]
p = Path(sys.argv[2])
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

echo "[$(date -u '+%F %T')] python: $PYTHON_BIN"
echo "[$(date -u '+%F %T')] batch_size=$BATCH_SIZE train_epochs=$TRAIN_EPOCHS min_free_gb=$MIN_FREE_GB"
echo "[$(date -u '+%F %T')] experiment=$EXPERIMENT"

for item in "${ERAS[@]}"; do
  IFS=':' read -r ERA_TAG TRAIN_START_D TRAIN_END_D <<<"$item"
  RUN_NAME="era_${ERA_TAG}_llama31_l6_v3"
  TRAIN_FILE="dataset/splits/train_${ERA_TAG}.csv"
  OUT_DIR="outputs/${RUN_NAME}_test_2022_2025"
  LOG_FILE="logs/${RUN_NAME}.log"

  if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "[$(date -u '+%F %T')] missing split: $TRAIN_FILE, skipping"
    continue
  fi
  if [[ ! -f "$TEST_FILE" ]]; then
    echo "[$(date -u '+%F %T')] missing test file: $TEST_FILE, skipping"
    continue
  fi

  if [[ "$(is_logged "$RUN_NAME")" == "yes" ]]; then
    echo "[$(date -u '+%F %T')] already logged: $RUN_NAME, skipping" | tee -a "$LOG_FILE"
    continue
  fi

  FREE_GB="$(check_free_space_gb)"
  if (( FREE_GB < MIN_FREE_GB )); then
    echo "[$(date -u '+%F %T')] low disk (${FREE_GB}G free < ${MIN_FREE_GB}G). stopping batch." | tee -a "$LOG_FILE"
    break
  fi

  echo "[$(date -u '+%F %T')] start training: $RUN_NAME (free=${FREE_GB}G)" | tee -a "$LOG_FILE"
  if ! "$PYTHON_BIN" run_main.py \
    --model_id "$RUN_NAME" \
    --data CRYPTEX \
    --root_path ./dataset \
    --data_path "splits/train_${ERA_TAG}.csv" \
    --train_epochs "$TRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seq_len 32 \
    --pred_len 8 \
    --llm_model LLAMA3.1 \
    --llm_layers 6 \
    --enc_in 5 \
    --num_workers 0 \
    --experiment_name "$EXPERIMENT" >>"$LOG_FILE" 2>&1; then
    echo "[$(date -u '+%F %T')] training failed: $RUN_NAME" | tee -a "$LOG_FILE"
    continue
  fi

  echo "[$(date -u '+%F %T')] start inference: $RUN_NAME" | tee -a "$LOG_FILE"
  if ! "$PYTHON_BIN" run_inference.py \
    --model_id "$RUN_NAME" \
    --llm_model LLAMA3.1 \
    --experiment_name "$EXPERIMENT" \
    --mlflow_tracking_uri sqlite:///mlflow.db \
    --data_path "./$TEST_FILE" \
    --save_path "./$OUT_DIR" >>"$LOG_FILE" 2>&1; then
    echo "[$(date -u '+%F %T')] inference failed: $RUN_NAME" | tee -a "$LOG_FILE"
    continue
  fi

  echo "[$(date -u '+%F %T')] update tracker: $RUN_NAME" | tee -a "$LOG_FILE"
  if ! "$PYTHON_BIN" utils/experiment_tracker.py \
    --experiment_name "$EXPERIMENT" \
    --run_name "$RUN_NAME" \
    --inference_csv "$OUT_DIR/inference.csv" \
    --train_start "$TRAIN_START_D" \
    --train_end "$TRAIN_END_D" \
    --test_start "$TEST_START" \
    --test_end "$TEST_END" \
    --notes "llama31_l6_v3, train=${ERA_TAG}, future_test=2022-2025" \
    --output_csv "$TRACKER_CSV" \
    --output_xlsx "$TRACKER_XLSX" \
    --log_inference_metrics_to_mlflow >>"$LOG_FILE" 2>&1; then
    echo "[$(date -u '+%F %T')] tracker update failed: $RUN_NAME" | tee -a "$LOG_FILE"
    continue
  fi

  echo "[$(date -u '+%F %T')] completed: $RUN_NAME" | tee -a "$LOG_FILE"
done

echo "[$(date -u '+%F %T')] llama31 4-year L6 v3 batch complete"
