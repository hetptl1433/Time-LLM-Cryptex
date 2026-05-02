#!/usr/bin/env python3
"""
Append/update an experiment tracking table (CSV/XLSX) using:
1) MLflow run metadata + internal train/val/test metrics
2) Future-test metrics computed from inference.csv

Usage example:
python3 utils/experiment_tracker.py \
  --experiment_name era_sensitivity \
  --run_name era_2019_2021 \
  --inference_csv outputs/era_2019_2021_test_2022_2026/inference.csv \
  --train_start 2019-01-01 --train_end 2021-12-31 \
  --test_start 2022-01-01 --test_end 2026-03-19
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment table tracker")
    parser.add_argument("--tracking_uri", default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--experiment_name", required=True, help="MLflow experiment name")
    parser.add_argument("--run_name", required=False, help="MLflow run name/tag (model_id)")
    parser.add_argument("--run_id", required=False, help="Exact MLflow run_id (optional)")

    parser.add_argument("--inference_csv", required=True, help="Path to inference.csv")
    parser.add_argument("--pred_col", default="close_predicted_1", help="Prediction column in inference CSV")
    parser.add_argument("--target_col", default="close", help="Target column in inference CSV")

    parser.add_argument("--train_start", required=True, help="Training window start YYYY-MM-DD")
    parser.add_argument("--train_end", required=True, help="Training window end YYYY-MM-DD")
    parser.add_argument("--test_start", required=True, help="Test window start YYYY-MM-DD")
    parser.add_argument("--test_end", required=True, help="Test window end YYYY-MM-DD")
    parser.add_argument("--notes", default="", help="Free-text notes")

    parser.add_argument("--output_csv", default="results/experiment_tracker.csv", help="Tracker CSV path")
    parser.add_argument("--output_xlsx", default="results/experiment_tracker.xlsx", help="Tracker XLSX path")
    parser.add_argument("--log_inference_metrics_to_mlflow", action="store_true", help="Also log inference metrics to MLflow run")
    return parser.parse_args()


def find_run(client: MlflowClient, experiment_name: str, run_id: Optional[str], run_name: Optional[str]):
    if run_id:
        return client.get_run(run_id)

    if not run_name:
        raise ValueError("Provide either --run_id or --run_name.")

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"Run '{run_name}' not found in experiment '{experiment_name}'.")
    return runs[0]


def compute_inference_metrics(inference_csv: Path, target_col: str, pred_col: str) -> Dict[str, float]:
    if not inference_csv.exists():
        raise FileNotFoundError(f"Inference CSV not found: {inference_csv}")

    df = pd.read_csv(inference_csv)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in {inference_csv}")
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}' in {inference_csv}")

    # 1-step alignment: prediction at t should match true target at t+1
    y_true = df[target_col].shift(-1)
    y_pred = df[pred_col]
    y_base = df[target_col]
    mask = y_true.notna() & y_pred.notna() & y_base.notna()
    if mask.sum() == 0:
        raise ValueError("No valid rows to evaluate in inference CSV after alignment/masking.")

    mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
    dir_acc = float(np.mean(np.sign(y_true[mask] - y_base[mask]) == np.sign(y_pred[mask] - y_base[mask])))
    return {
        "infer_rows_used": int(mask.sum()),
        "infer_mae_future": mae,
        "infer_rmse_future": rmse,
        "infer_dir_acc_future": dir_acc,
    }


def safe_metric(metrics: Dict[str, float], key: str):
    return metrics.get(key, np.nan)


def safe_param(params: Dict[str, str], key: str):
    return params.get(key, "")


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    run = find_run(client, args.experiment_name, args.run_id, args.run_name)
    run_id = run.info.run_id
    run_name = run.data.tags.get("mlflow.runName", "")
    metrics = run.data.metrics
    params = run.data.params

    inference_metrics = compute_inference_metrics(
        inference_csv=Path(args.inference_csv),
        target_col=args.target_col,
        pred_col=args.pred_col,
    )

    row = {
        "logged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": args.experiment_name,
        "run_id": run_id,
        "run_name": run_name,
        "status": run.info.status,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "train_data_path": safe_param(params, "data_path"),
        "seq_len": safe_param(params, "seq_len"),
        "pred_len": safe_param(params, "pred_len"),
        "train_epochs": safe_param(params, "train_epochs"),
        "batch_size": safe_param(params, "batch_size"),
        "llm_model": safe_param(params, "llm_model"),
        "llm_layers": safe_param(params, "llm_layers"),
        "learning_rate": safe_param(params, "learning_rate"),
        "features": safe_param(params, "features"),
        "target": safe_param(params, "target"),
        "train_mse_loss": safe_metric(metrics, "train_mse_loss"),
        "vali_mse_loss": safe_metric(metrics, "vali_mse_loss"),
        "vali_mae_metric": safe_metric(metrics, "vali_mae_metric"),
        "test_mse_loss_internal": safe_metric(metrics, "test_mse_loss"),
        "test_mae_metric_internal": safe_metric(metrics, "test_mae_metric"),
        **inference_metrics,
        "notes": args.notes,
    }

    output_csv = Path(args.output_csv)
    output_xlsx = Path(args.output_xlsx)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        tracker_df = pd.read_csv(output_csv)
    else:
        tracker_df = pd.DataFrame()

    # Clean up any accidental fully-empty rows.
    tracker_df = tracker_df.dropna(how="all").copy()

    # Ensure tracker has all expected columns before upsert.
    for col in row.keys():
        if col not in tracker_df.columns:
            tracker_df[col] = np.nan

    # Upsert by run_id (so reruns update the same row instead of duplicating)
    run_id_series = tracker_df.get("run_id", pd.Series(dtype=str)).astype(str)
    match_mask = run_id_series == str(run_id)
    if match_mask.any():
        for key, value in row.items():
            tracker_df.loc[match_mask, key] = value
    else:
        tracker_df = pd.concat([tracker_df, pd.DataFrame([row])], ignore_index=True)

    tracker_df.to_csv(output_csv, index=False)

    # Best effort XLSX output for direct Excel opening.
    try:
        tracker_df.to_excel(output_xlsx, index=False)
        xlsx_msg = f" and {output_xlsx}"
    except Exception:
        xlsx_msg = " (XLSX export skipped; install openpyxl if needed)"

    if args.log_inference_metrics_to_mlflow:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                {
                    "infer_mae_future": inference_metrics["infer_mae_future"],
                    "infer_rmse_future": inference_metrics["infer_rmse_future"],
                    "infer_dir_acc_future": inference_metrics["infer_dir_acc_future"],
                }
            )

    print(f"Tracker updated: {output_csv}{xlsx_msg}")
    print(
        "Future-test metrics:",
        {
            "infer_mae_future": inference_metrics["infer_mae_future"],
            "infer_rmse_future": inference_metrics["infer_rmse_future"],
            "infer_dir_acc_future": inference_metrics["infer_dir_acc_future"],
            "infer_rows_used": inference_metrics["infer_rows_used"],
        },
    )


if __name__ == "__main__":
    main()
