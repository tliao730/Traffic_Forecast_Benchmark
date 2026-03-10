import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def load_and_align(csv_path: str, pred_stat: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', errors='coerce')
    
    # Convert value column to numeric
    df["value"] = pd.to_numeric(df["value"], errors='coerce')

    truth = df[df["stat"] == "truth"].copy()
    if pred_stat not in df["stat"].unique():
        pred_stat = "mean"
    pred = df[df["stat"] == pred_stat].copy()

    key_cols = ["ds_config", "sample_idx", "item_id", "timestamp", "dim"]
    merged = truth.merge(
        pred,
        on=key_cols,
        how="inner",
        suffixes=("_truth", "_pred"),
    )

    merged["abs_error"] = (merged["value_pred"] - merged["value_truth"]).abs()
    return merged


def compute_time_errors(merged: pd.DataFrame) -> pd.DataFrame:
    time_errors = (
        merged.groupby("timestamp", as_index=False)["abs_error"]
        .mean()
        .sort_values("timestamp")
    )
    time_errors.rename(columns={"abs_error": "mean_abs_error"}, inplace=True)
    return time_errors


def compute_id_errors(merged: pd.DataFrame) -> pd.DataFrame:
    id_errors = (
        merged.groupby("item_id", as_index=False)["abs_error"]
        .mean()
        .sort_values("abs_error", ascending=False)
    )
    id_errors.rename(columns={"abs_error": "mean_abs_error"}, inplace=True)
    return id_errors


def compute_hourly_errors(merged: pd.DataFrame) -> pd.DataFrame:
    hourly = merged.copy()
    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly_errors = (
        hourly.groupby("hour", as_index=False)["abs_error"]
        .mean()
        .sort_values("hour")
    )
    hourly_errors.rename(columns={"abs_error": "mean_abs_error"}, inplace=True)
    return hourly_errors


def compute_time_of_day_errors(merged: pd.DataFrame) -> pd.DataFrame:
    tod = merged.copy()
    tod["hour"] = tod["timestamp"].dt.hour
    tod["minute"] = tod["timestamp"].dt.minute
    tod_errors = (
        tod.groupby(["hour", "minute"], as_index=False)["abs_error"]
        .mean()
        .sort_values(["hour", "minute"])
    )
    tod_errors["slot"] = tod_errors["hour"] * 4 + (tod_errors["minute"] // 15)
    tod_errors.rename(columns={"abs_error": "mean_abs_error"}, inplace=True)
    return tod_errors


def plot_time_errors(time_errors: pd.DataFrame, out_path: str, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(time_errors["timestamp"], time_errors["mean_abs_error"], linewidth=1.2)
    plt.title(title)
    plt.xlabel("timestamp")
    plt.ylabel("mean abs error")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hourly_errors(hourly_errors: pd.DataFrame, out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(hourly_errors["hour"], hourly_errors["mean_abs_error"], width=0.8)
    plt.title(title)
    plt.xlabel("hour of day")
    plt.ylabel("mean abs error")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_time_of_day_errors(tod_errors: pd.DataFrame, out_path: str, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(tod_errors["slot"], tod_errors["mean_abs_error"], linewidth=1.2)
    plt.title(title)
    plt.xlabel("time-of-day slot (0-95, 15-min intervals)")
    plt.ylabel("mean abs error")
    plt.xticks(range(0, 96, 8))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def analyze(csv_path: str, pred_stat: str, output_dir: str, top_k: int) -> Tuple[str, str, str, str, str]:
    os.makedirs(output_dir, exist_ok=True)

    merged = load_and_align(csv_path, pred_stat)

    time_errors = compute_time_errors(merged)
    id_errors = compute_id_errors(merged)
    hourly_errors = compute_hourly_errors(merged)
    tod_errors = compute_time_of_day_errors(merged)

    time_csv = os.path.join(output_dir, "time_mean_abs_error.csv")
    hourly_csv = os.path.join(output_dir, "hourly_mean_abs_error.csv")
    id_csv = os.path.join(output_dir, "id_mean_abs_error.csv")
    top_csv = os.path.join(output_dir, f"top_{top_k}_hardest_ids.csv")
    tod_csv = os.path.join(output_dir, "tod_mean_abs_error.csv")

    time_errors.to_csv(time_csv, index=False)
    hourly_errors.to_csv(hourly_csv, index=False)
    id_errors.to_csv(id_csv, index=False)
    id_errors.head(top_k).to_csv(top_csv, index=False)
    tod_errors.to_csv(tod_csv, index=False)

    plot_path = os.path.join(output_dir, "time_mean_abs_error.png")
    title = f"Mean Abs Error by Time ({pred_stat})"
    plot_time_errors(time_errors, plot_path, title)

    hourly_plot_path = os.path.join(output_dir, "hourly_mean_abs_error.png")
    hourly_title = f"Mean Abs Error by Hour ({pred_stat})"
    plot_hourly_errors(hourly_errors, hourly_plot_path, hourly_title)

    tod_plot_path = os.path.join(output_dir, "tod_mean_abs_error.png")
    tod_title = f"Mean Abs Error by Time-of-Day (15-min) ({pred_stat})"
    plot_time_of_day_errors(tod_errors, tod_plot_path, tod_title)

    return time_csv, hourly_csv, id_csv, top_csv, tod_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to predictions CSV")
    parser.add_argument(
        "--pred-stat",
        default="mean",
        help="prediction stat to compare against truth (default: mean)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="directory to save analysis outputs",
    )
    parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()
    time_csv, hourly_csv, id_csv, top_csv, tod_csv = analyze(
        args.csv, args.pred_stat, args.output_dir, args.top_k
    )

    print("Saved:")
    print(f"  {time_csv}")
    print(f"  {hourly_csv}")
    print(f"  {id_csv}")
    print(f"  {top_csv}")
    print(f"  {tod_csv}")
