#!/usr/bin/env python3
"""
Create charts from the experiment tracker data in results/.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


FOUR_YEAR_RUNS = {
    "era_2013_2016",
    "era_2014_2017",
    "era_2015_2018",
    "era_2016_2019",
    "era_2017_2020",
    "era_2018_2021",
}

METRICS = [
    ("infer_mae_future", "Future MAE", "MAE"),
    ("infer_rmse_future", "Future RMSE", "RMSE"),
    ("infer_dir_acc_future", "Future Directional Accuracy", "Directional Accuracy"),
]

COLORS = {
    "4-year blocks": "#1f77b4",
    "Growing windows": "#d62728",
}


def load_tracker() -> pd.DataFrame:
    presentation_xlsx = Path("results/experiment_tracker_presentation.xlsx")
    raw_xlsx = Path("results/experiment_tracker.xlsx")

    if presentation_xlsx.exists():
        df = pd.read_excel(presentation_xlsx, sheet_name="Ordered Runs")
    elif raw_xlsx.exists():
        df = pd.read_excel(raw_xlsx)
        for col in ["train_start", "train_end", "test_start", "test_end"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        df["train_years"] = ((df["train_end"] - df["train_start"]).dt.days + 1) / 365.25
        df["group"] = df["run_name"].apply(
            lambda name: "4-year blocks" if name in FOUR_YEAR_RUNS else "Growing windows"
        )
    else:
        raise FileNotFoundError("No experiment tracker workbook found in results/.")

    required = {"group", "run_name", "train_years", "infer_mae_future", "infer_rmse_future", "infer_dir_acc_future"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["train_years"] = pd.to_numeric(df["train_years"], errors="coerce")
    for metric, _, _ in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.dropna(subset=["group", "run_name", "train_years", *[metric for metric, _, _ in METRICS]])
    df["group"] = pd.Categorical(df["group"], categories=["4-year blocks", "Growing windows"], ordered=True)
    return df.sort_values(["group", "train_years", "run_name"]).reset_index(drop=True)


def style_axis(ax: plt.Axes, metric: str, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if metric == "infer_dir_acc_future":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def plot_by_run(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    x = range(len(df))

    for ax, (metric, title, ylabel) in zip(axes, METRICS):
        for group, group_df in df.groupby("group", observed=True):
            idx = group_df.index.to_list()
            ax.plot(idx, group_df[metric], marker="o", linewidth=2, label=group, color=COLORS[str(group)])
        style_axis(ax, metric, title, ylabel)

    axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(df["run_name"], rotation=45, ha="right")
    axes[-1].set_xlabel("Run Name")
    fig.suptitle("Experiment Tracker Metrics by Run", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_by_train_years(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for ax, (metric, title, ylabel) in zip(axes, METRICS):
        for group, group_df in df.groupby("group", observed=True):
            ordered = group_df.sort_values(["train_years", "run_name"])
            ax.plot(
                ordered["train_years"],
                ordered[metric],
                marker="o",
                linewidth=2,
                label=group,
                color=COLORS[str(group)],
            )
        style_axis(ax, metric, f"{title} vs Train Years", ylabel)

    axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[-1].set_xlabel("Training Window Length (years)")
    fig.suptitle("Experiment Tracker Metrics vs Training Window", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_tracker()
    out_dir = Path("results/charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_run = out_dir / "metrics_by_run.png"
    by_years = out_dir / "metrics_by_train_years.png"

    plot_by_run(df, by_run)
    plot_by_train_years(df, by_years)

    print(f"Wrote: {by_run}")
    print(f"Wrote: {by_years}")


if __name__ == "__main__":
    main()
