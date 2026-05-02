#!/usr/bin/env python3
"""
Build a presentation-friendly experiment tracker from results/experiment_tracker.csv.
Keeps the original tracker intact and writes new ordered CSV/XLSX files.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


FOUR_YEAR_RUNS = {
    "era_2013_2016",
    "era_2014_2017",
    "era_2015_2018",
    "era_2016_2019",
    "era_2017_2020",
    "era_2018_2021",
}


def main() -> None:
    src = Path("results/experiment_tracker.csv")
    if not src.exists():
        raise FileNotFoundError(f"Missing source tracker: {src}")

    df = pd.read_csv(src).copy()
    if df.empty:
        raise ValueError("Source tracker is empty.")

    for c in ["train_start", "train_end", "test_start", "test_end"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["train_years"] = ((df["train_end"] - df["train_start"]).dt.days + 1) / 365.25
    df["train_window"] = df["train_start"].dt.strftime("%Y-%m-%d") + " to " + df["train_end"].dt.strftime("%Y-%m-%d")
    df["test_window"] = df["test_start"].dt.strftime("%Y-%m-%d") + " to " + df["test_end"].dt.strftime("%Y-%m-%d")
    df["group"] = df["run_name"].apply(lambda x: "4-year blocks" if x in FOUR_YEAR_RUNS else "Growing windows")

    four = df[df["group"] == "4-year blocks"].sort_values("train_start")
    grow = df[df["group"] == "Growing windows"].sort_values(["train_years", "train_start"])
    out = pd.concat([four, grow], ignore_index=True)

    # Ranks across all runs
    out["rank_mae"] = out["infer_mae_future"].rank(method="min", ascending=True).astype("Int64")
    out["rank_rmse"] = out["infer_rmse_future"].rank(method="min", ascending=True).astype("Int64")
    out["rank_dir_acc"] = out["infer_dir_acc_future"].rank(method="min", ascending=False).astype("Int64")

    cols = [
        "group",
        "run_name",
        "train_window",
        "train_years",
        "test_window",
        "infer_mae_future",
        "infer_rmse_future",
        "infer_dir_acc_future",
        "rank_mae",
        "rank_rmse",
        "rank_dir_acc",
    ]
    present = out[cols].copy()
    present["train_years"] = present["train_years"].round(2)

    summary = pd.DataFrame(
        [
            {
                "metric": "Best MAE (lower is better)",
                "run_name": out.loc[out["infer_mae_future"].idxmin(), "run_name"],
                "value": float(out["infer_mae_future"].min()),
            },
            {
                "metric": "Best RMSE (lower is better)",
                "run_name": out.loc[out["infer_rmse_future"].idxmin(), "run_name"],
                "value": float(out["infer_rmse_future"].min()),
            },
            {
                "metric": "Best Directional Accuracy (higher is better)",
                "run_name": out.loc[out["infer_dir_acc_future"].idxmax(), "run_name"],
                "value": float(out["infer_dir_acc_future"].max()),
            },
        ]
    )

    out_csv = Path("results/experiment_tracker_presentation.csv")
    out_xlsx = Path("results/experiment_tracker_presentation.xlsx")

    present.to_csv(out_csv, index=False)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        present.to_excel(writer, sheet_name="Ordered Runs", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

        ws = writer.book["Ordered Runs"]
        ws.freeze_panes = "A2"
        widths = {
            "A": 20,
            "B": 18,
            "C": 28,
            "D": 12,
            "E": 28,
            "F": 18,
            "G": 18,
            "H": 18,
            "I": 10,
            "J": 10,
            "K": 12,
        }
        for col, w in widths.items():
            ws.column_dimensions[col].width = w

        ws2 = writer.book["Summary"]
        ws2.freeze_panes = "A2"
        ws2.column_dimensions["A"].width = 42
        ws2.column_dimensions["B"].width = 18
        ws2.column_dimensions["C"].width = 16

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_xlsx}")


if __name__ == "__main__":
    main()
