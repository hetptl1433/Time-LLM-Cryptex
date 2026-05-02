#!/usr/bin/env python3
"""
Build a segmented experiment workbook:
- Four-year block results, ranks, analysis
- Growing-window results, ranks, analysis
All in one Excel file with separate sheets.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


FOUR_YEAR_RUNS = [
    "era_2013_2016",
    "era_2014_2017",
    "era_2015_2018",
    "era_2016_2019",
    "era_2017_2020",
    "era_2018_2021",
]

GROWING_RUNS = [
    "era_2021_2021",
    "era_2020_2021",
    "era_2019_2021",
    "era_2018_2021",
    "era_2017_2021",
    "era_2016_2021",
    "era_2015_2021",
    "era_2014_2021",
    "era_2013_2021",
]


def prep_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out["train_years"] = ((out["train_end"] - out["train_start"]).dt.days + 1) / 365.25
    out["train_window"] = out["train_start"].dt.strftime("%Y-%m-%d") + " to " + out["train_end"].dt.strftime("%Y-%m-%d")
    out["test_window"] = out["test_start"].dt.strftime("%Y-%m-%d") + " to " + out["test_end"].dt.strftime("%Y-%m-%d")
    return out


def result_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "run_name",
        "train_window",
        "train_years",
        "test_window",
        "infer_mae_future",
        "infer_rmse_future",
        "infer_dir_acc_future",
    ]
    t = df[cols].copy()
    t["train_years"] = t["train_years"].round(2)
    return t


def rank_table(df: pd.DataFrame) -> pd.DataFrame:
    t = result_table(df).copy()
    t["rank_mae"] = t["infer_mae_future"].rank(method="min", ascending=True).astype("Int64")
    t["rank_rmse"] = t["infer_rmse_future"].rank(method="min", ascending=True).astype("Int64")
    t["rank_dir_acc"] = t["infer_dir_acc_future"].rank(method="min", ascending=False).astype("Int64")
    return t.sort_values(["rank_mae", "rank_rmse", "rank_dir_acc", "run_name"])


def analysis_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    # Summary rows
    best_mae_idx = df["infer_mae_future"].idxmin()
    best_rmse_idx = df["infer_rmse_future"].idxmin()
    best_da_idx = df["infer_dir_acc_future"].idxmax()
    worst_mae_idx = df["infer_mae_future"].idxmax()
    worst_rmse_idx = df["infer_rmse_future"].idxmax()
    worst_da_idx = df["infer_dir_acc_future"].idxmin()

    rows = [
        {"metric": f"{label}: Best MAE (lower better)", "run_name": df.loc[best_mae_idx, "run_name"], "value": float(df.loc[best_mae_idx, "infer_mae_future"])},
        {"metric": f"{label}: Best RMSE (lower better)", "run_name": df.loc[best_rmse_idx, "run_name"], "value": float(df.loc[best_rmse_idx, "infer_rmse_future"])},
        {"metric": f"{label}: Best DirAcc (higher better)", "run_name": df.loc[best_da_idx, "run_name"], "value": float(df.loc[best_da_idx, "infer_dir_acc_future"])},
        {"metric": f"{label}: Worst MAE", "run_name": df.loc[worst_mae_idx, "run_name"], "value": float(df.loc[worst_mae_idx, "infer_mae_future"])},
        {"metric": f"{label}: Worst RMSE", "run_name": df.loc[worst_rmse_idx, "run_name"], "value": float(df.loc[worst_rmse_idx, "infer_rmse_future"])},
        {"metric": f"{label}: Worst DirAcc", "run_name": df.loc[worst_da_idx, "run_name"], "value": float(df.loc[worst_da_idx, "infer_dir_acc_future"])},
        {"metric": f"{label}: Mean MAE", "run_name": "-", "value": float(df["infer_mae_future"].mean())},
        {"metric": f"{label}: Mean RMSE", "run_name": "-", "value": float(df["infer_rmse_future"].mean())},
        {"metric": f"{label}: Mean DirAcc", "run_name": "-", "value": float(df["infer_dir_acc_future"].mean())},
    ]

    # Simple trend for growing windows (how metrics move with train_years)
    if len(df) >= 3 and "train_years" in df.columns:
        corr_mae = float(df["train_years"].corr(df["infer_mae_future"]))
        corr_rmse = float(df["train_years"].corr(df["infer_rmse_future"]))
        corr_da = float(df["train_years"].corr(df["infer_dir_acc_future"]))
        rows.extend(
            [
                {"metric": f"{label}: Corr(train_years, MAE)", "run_name": "-", "value": corr_mae},
                {"metric": f"{label}: Corr(train_years, RMSE)", "run_name": "-", "value": corr_rmse},
                {"metric": f"{label}: Corr(train_years, DirAcc)", "run_name": "-", "value": corr_da},
            ]
        )

    return pd.DataFrame(rows)


def set_widths(ws, widths: dict[str, int]) -> None:
    ws.freeze_panes = "A2"
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


def main() -> None:
    src = Path("results/experiment_tracker.csv")
    if not src.exists():
        raise FileNotFoundError(f"Missing source tracker: {src}")

    full = prep_base(pd.read_csv(src))
    if full.empty:
        raise ValueError("Source tracker is empty.")

    four = full[full["run_name"].isin(FOUR_YEAR_RUNS)].copy()
    four["order"] = four["run_name"].apply(lambda x: FOUR_YEAR_RUNS.index(x))
    four = four.sort_values("order").drop(columns=["order"])

    grow = full[full["run_name"].isin(GROWING_RUNS)].copy()
    grow["order"] = grow["run_name"].apply(lambda x: GROWING_RUNS.index(x))
    grow = grow.sort_values("order").drop(columns=["order"])

    out_xlsx = Path("results/experiment_tracker_segmented.xlsx")
    out_csv_four = Path("results/experiment_tracker_four_year.csv")
    out_csv_grow = Path("results/experiment_tracker_growing.csv")

    four_results = result_table(four)
    grow_results = result_table(grow)
    four_ranks = rank_table(four)
    grow_ranks = rank_table(grow)
    four_analysis = analysis_table(four, "4-year")
    grow_analysis = analysis_table(grow, "growing")

    four_results.to_csv(out_csv_four, index=False)
    grow_results.to_csv(out_csv_grow, index=False)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        four_results.to_excel(writer, sheet_name="4Y_Results", index=False)
        four_ranks.to_excel(writer, sheet_name="4Y_Ranks", index=False)
        four_analysis.to_excel(writer, sheet_name="4Y_Analysis", index=False)
        grow_results.to_excel(writer, sheet_name="Growing_Results", index=False)
        grow_ranks.to_excel(writer, sheet_name="Growing_Ranks", index=False)
        grow_analysis.to_excel(writer, sheet_name="Growing_Analysis", index=False)

        set_widths(writer.book["4Y_Results"], {"A": 18, "B": 28, "C": 12, "D": 28, "E": 18, "F": 18, "G": 18})
        set_widths(writer.book["4Y_Ranks"], {"A": 18, "B": 28, "C": 12, "D": 28, "E": 18, "F": 18, "G": 18, "H": 10, "I": 10, "J": 12})
        set_widths(writer.book["4Y_Analysis"], {"A": 42, "B": 18, "C": 16})
        set_widths(writer.book["Growing_Results"], {"A": 18, "B": 28, "C": 12, "D": 28, "E": 18, "F": 18, "G": 18})
        set_widths(writer.book["Growing_Ranks"], {"A": 18, "B": 28, "C": 12, "D": 28, "E": 18, "F": 18, "G": 18, "H": 10, "I": 10, "J": 12})
        set_widths(writer.book["Growing_Analysis"], {"A": 42, "B": 18, "C": 16})

    print(f"Wrote: {out_xlsx}")
    print(f"Wrote: {out_csv_four}")
    print(f"Wrote: {out_csv_grow}")


if __name__ == "__main__":
    main()
