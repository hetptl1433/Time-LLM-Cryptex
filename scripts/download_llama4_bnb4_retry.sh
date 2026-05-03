#!/usr/bin/env bash
set -u

REPO="bnb-community/Llama-4-Scout-17B-16E-Instruct-bnb-4bit"
TARGET_DIR="/nfs_nvme/het/Time-LLM-Cryptex/llm_weights/Llama-4-Scout-17B-16E-Instruct-bnb-4bit"
LOG_FILE="/nfs_nvme/het/Time-LLM-Cryptex/logs/llama4_scout_bnb4_download.log"

mkdir -p "$(dirname "$LOG_FILE")" "$TARGET_DIR"

ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  echo "[$(date -u '+%F %T')] attempt=$ATTEMPT start" | tee -a "$LOG_FILE"

  # Resume-capable, single worker for stability.
  hf download "$REPO" \
    --local-dir "$TARGET_DIR" \
    --max-workers 1 >> "$LOG_FILE" 2>&1
  RC=$?

  if [ $RC -eq 0 ]; then
    echo "[$(date -u '+%F %T')] download completed" | tee -a "$LOG_FILE"
    break
  fi

  SIZE=$(du -sh "$TARGET_DIR" 2>/dev/null | awk '{print $1}')
  echo "[$(date -u '+%F %T')] attempt=$ATTEMPT failed rc=$RC size=$SIZE; retry in 20s" | tee -a "$LOG_FILE"
  sleep 20
done
