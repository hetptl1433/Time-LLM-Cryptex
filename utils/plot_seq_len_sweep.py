#!/usr/bin/env python3
"""
Rebuild the fixed-window sequence-length sweep chart from the tracker workbook.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("infer_mae_future", "Future MAE"),
    ("infer_rmse_future", "Future RMSE"),
    ("infer_dir_acc_future", "Directional Accuracy"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sequence-length sweep results")
    parser.add_argument(
        "--tracker_csv",
        default="results/experiment_tracker_llama31_l6_seq.csv",
        help="Tracker CSV path",
    )
    parser.add_argument(
        "--tracker_xlsx",
        default="results/experiment_tracker_llama31_l6_seq.xlsx",
        help="Tracker XLSX path (used if CSV is missing)",
    )
    parser.add_argument(
        "--run_prefix",
        default="era_2018_2021_llama31_l6_seq",
        help="Only include runs whose run_name starts with this prefix",
    )
    parser.add_argument(
        "--output_stem",
        default="seq_sweep_llama31_l6_2018_2021",
        help="Base filename for output artifacts",
    )
    parser.add_argument(
        "--output_dir",
        default="results/charts/seq_len",
        help="Directory for the output artifacts",
    )
    return parser.parse_args()


def load_tracker(csv_path: Path, xlsx_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    raise FileNotFoundError(f"Neither tracker file exists: {csv_path} or {xlsx_path}")


def prepare_sweep(df: pd.DataFrame, run_prefix: str) -> pd.DataFrame:
    required = {"run_name", "seq_len", *[metric for metric, _ in METRICS]}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    sweep = df.copy()
    sweep["run_name"] = sweep["run_name"].astype(str)
    sweep = sweep[sweep["run_name"].str.startswith(run_prefix)].copy()

    if "status" in sweep.columns:
        sweep["status"] = sweep["status"].astype(str)
        sweep = sweep[sweep["status"].eq("FINISHED")].copy()

    sweep["seq_len"] = pd.to_numeric(sweep["seq_len"], errors="coerce")
    for metric, _ in METRICS:
        sweep[metric] = pd.to_numeric(sweep[metric], errors="coerce")

    sweep = sweep.dropna(subset=["seq_len", *[metric for metric, _ in METRICS]]).copy()
    if sweep.empty:
        raise ValueError(f"No completed sweep rows found for run_prefix='{run_prefix}'.")

    if "logged_at_utc" in sweep.columns:
        sweep["logged_at_utc"] = pd.to_datetime(sweep["logged_at_utc"], errors="coerce", utc=True)
        sweep = sweep.sort_values(["seq_len", "logged_at_utc", "run_name"]).groupby("seq_len", as_index=False).tail(1)
    else:
        sweep = sweep.sort_values(["seq_len", "run_name"]).drop_duplicates(subset=["seq_len"], keep="last")

    return sweep.sort_values("seq_len").reset_index(drop=True)


def year_span(start_value: object, end_value: object) -> str | None:
    start = pd.to_datetime(start_value, errors="coerce")
    end = pd.to_datetime(end_value, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    return f"{start.year}-{end.year}"


def build_title(sweep: pd.DataFrame) -> str:
    model_name = "Sequence Length Sweep"
    llm_model = sweep["llm_model"].dropna().astype(str).unique().tolist() if "llm_model" in sweep.columns else []
    llm_layers = sweep["llm_layers"].dropna().astype(str).unique().tolist() if "llm_layers" in sweep.columns else []
    if len(llm_model) == 1 and len(llm_layers) == 1:
        model_name = f"Sequence Length Sweep ({llm_model[0]}, {llm_layers[0]} Layers)"

    subtitle = None
    if {"train_start", "train_end", "test_start", "test_end"}.issubset(sweep.columns):
        train_span = year_span(sweep["train_start"].iloc[0], sweep["train_end"].iloc[0])
        test_span = year_span(sweep["test_start"].iloc[0], sweep["test_end"].iloc[0])
        if train_span and test_span:
            subtitle = f"Train: {train_span} | Test: {test_span}"

    return model_name if subtitle is None else f"{model_name}\n{subtitle}"


def plot_sweep(sweep: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    x_values = sweep["seq_len"]

    for ax, (metric, ylabel) in zip(axes, METRICS):
        ax.plot(x_values, sweep[metric], marker="o", linewidth=1.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Sequence Length")
    axes[-1].set_xticks(x_values.tolist())
    fig.suptitle(build_title(sweep), fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tracker = load_tracker(Path(args.tracker_csv), Path(args.tracker_xlsx))
    sweep = prepare_sweep(tracker, args.run_prefix)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points_csv = output_dir / f"{args.output_stem}_points.csv"
    output_png = output_dir / f"{args.output_stem}.png"
    output_pdf = output_dir / f"{args.output_stem}.pdf"

    points = sweep[["seq_len", *[metric for metric, _ in METRICS]]].copy()
    points.to_csv(points_csv, index=False)
    plot_sweep(sweep, output_png, output_pdf)

    print(f"Wrote: {points_csv}")
    print(f"Wrote: {output_png}")
    print(f"Wrote: {output_pdf}")
    print(f"Points included: {len(points)}")


if __name__ == "__main__":
    main()
