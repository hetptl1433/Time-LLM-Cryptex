#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def main() -> None:
    src = Path("results/experiment_tracker_llama31.csv")
    if not src.exists():
        raise FileNotFoundError(f"Missing tracker: {src}")

    df = pd.read_csv(src)
    if df.empty or "run_name" not in df.columns:
        raise ValueError("Tracker is empty or missing run_name.")

    pat = re.compile(r"^era_(\d{4})_2021_llama31$")
    grow = df[df["run_name"].astype(str).str.match(pat)].copy()
    if grow.empty:
        raise ValueError("No growing-window LLAMA3.1 runs found (era_YYYY_2021_llama31).")

    for c in ["train_start", "train_end", "test_start", "test_end"]:
        grow[c] = pd.to_datetime(grow[c], errors="coerce")

    for c in ["infer_mae_future", "infer_rmse_future", "infer_dir_acc_future"]:
        grow[c] = pd.to_numeric(grow[c], errors="coerce")

    grow["start_year"] = grow["run_name"].str.extract(r"era_(\d{4})_2021_llama31").astype(int)
    grow["end_year"] = 2021
    grow["window_years"] = grow["end_year"] - grow["start_year"] + 1
    grow["train_window"] = (
        grow["train_start"].dt.strftime("%Y-%m-%d")
        + " to "
        + grow["train_end"].dt.strftime("%Y-%m-%d")
    )

    out_cols = [
        "run_name",
        "window_years",
        "train_window",
        "infer_mae_future",
        "infer_rmse_future",
        "infer_dir_acc_future",
    ]
    out = grow[out_cols].sort_values("window_years").reset_index(drop=True)

    out["rank_mae"] = out["infer_mae_future"].rank(method="min", ascending=True).astype("Int64")
    out["rank_rmse"] = out["infer_rmse_future"].rank(method="min", ascending=True).astype("Int64")
    out["rank_dir_acc"] = out["infer_dir_acc_future"].rank(method="min", ascending=False).astype("Int64")

    best_mae = out.loc[out["infer_mae_future"].idxmin()]
    best_rmse = out.loc[out["infer_rmse_future"].idxmin()]
    best_dir = out.loc[out["infer_dir_acc_future"].idxmax()]

    summary = pd.DataFrame(
        [
            {
                "best_by": "MAE (lower better)",
                "run_name": best_mae["run_name"],
                "window_years": int(best_mae["window_years"]),
                "value": float(best_mae["infer_mae_future"]),
            },
            {
                "best_by": "RMSE (lower better)",
                "run_name": best_rmse["run_name"],
                "window_years": int(best_rmse["window_years"]),
                "value": float(best_rmse["infer_rmse_future"]),
            },
            {
                "best_by": "Directional Accuracy (higher better)",
                "run_name": best_dir["run_name"],
                "window_years": int(best_dir["window_years"]),
                "value": float(best_dir["infer_dir_acc_future"]),
            },
        ]
    )

    out_csv = Path("results/llama31_growing_windows_ranked.csv")
    out_xlsx = Path("results/llama31_growing_windows_ranked.xlsx")

    out.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="GrowingWindows", index=False)
        summary.to_excel(writer, sheet_name="BestWindow", index=False)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_xlsx}")
    print("\nBest windows:")
    for _, row in summary.iterrows():
        print(
            f"- {row['best_by']}: {row['run_name']} (years={int(row['window_years'])}, value={row['value']})"
        )


if __name__ == "__main__":
    main()
