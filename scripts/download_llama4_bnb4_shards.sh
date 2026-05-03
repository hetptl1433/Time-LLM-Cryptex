#!/usr/bin/env bash
set -euo pipefail

REPO="bnb-community/Llama-4-Scout-17B-16E-Instruct-bnb-4bit"
TARGET_DIR="/nfs_nvme/het/Time-LLM-Cryptex/llm_weights/Llama-4-Scout-17B-16E-Instruct-bnb-4bit"
LOG_FILE="/nfs_nvme/het/Time-LLM-Cryptex/logs/llama4_scout_bnb4_download.log"

mkdir -p "$TARGET_DIR" "$(dirname "$LOG_FILE")"

SMALL_FILES=(
  "config.json"
  "generation_config.json"
  "README.md"
  "chat_template.json"
  "model.safetensors.index.json"
)

for f in "${SMALL_FILES[@]}"; do
  if [[ ! -f "$TARGET_DIR/$f" ]]; then
    echo "[$(date -u '+%F %T')] downloading $f" | tee -a "$LOG_FILE"
    hf download "$REPO" "$f" --local-dir "$TARGET_DIR" >> "$LOG_FILE" 2>&1
  fi
done

for i in $(seq -w 1 12); do
  shard="model-000${i}-of-00012.safetensors"
  if [[ ! -f "$TARGET_DIR/$shard" ]]; then
    echo "[$(date -u '+%F %T')] downloading $shard" | tee -a "$LOG_FILE"
    while true; do
      if hf download "$REPO" "$shard" --local-dir "$TARGET_DIR" >> "$LOG_FILE" 2>&1; then
        break
      fi
      echo "[$(date -u '+%F %T')] retry $shard in 20s" | tee -a "$LOG_FILE"
      sleep 20
    done
  fi
  echo "[$(date -u '+%F %T')] have $shard" | tee -a "$LOG_FILE"
done

echo "[$(date -u '+%F %T')] ALL SHARDS COMPLETE" | tee -a "$LOG_FILE"
