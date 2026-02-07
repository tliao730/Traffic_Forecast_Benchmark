from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sd" / "sd_meta.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--err-path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "astgcn" / "id_mean_abs_error.csv"),
        help="Path to id_mean_abs_error.csv",
    )
    parser.add_argument(
        "--region-col",
        type=str,
        default="County",
        help="Region column in sd_meta.csv (e.g., County, District)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="sd_region_mean_abs_error.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default="sd_region_error.pdf",
        help="Output PDF filename",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation function for errors",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=0,
        help="Number of discrete levels to bucket errors (0 = no bucketing)",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for level colors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    err_path = Path(args.err_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    err_df = pd.read_csv(err_path)
    err_df["item_id"] = err_df["item_id"].astype(int)

    if args.region_col not in df.columns:
        raise ValueError(f"Region column not found: {args.region_col}")

    merged = df.merge(err_df, left_on="ID", right_on="item_id", how="inner")
    if merged.empty:
        raise ValueError("No matching IDs found between meta and error CSV.")

    if args.agg == "mean":
        region_err = merged.groupby(args.region_col, as_index=False)["mean_abs_error"].mean()
    else:
        region_err = merged.groupby(args.region_col, as_index=False)["mean_abs_error"].median()

    region_err = region_err.sort_values("mean_abs_error", ascending=False)

    if args.levels and args.levels > 1:
        region_err["level"] = pd.qcut(
            region_err["mean_abs_error"],
            q=args.levels,
            labels=[f"L{i+1}" for i in range(args.levels)],
        )
    out_csv = out_dir / args.out_csv
    region_err.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    if args.levels and args.levels > 1:
        cmap = plt.get_cmap(args.palette, args.levels)
        colors = [cmap(i) for i in range(args.levels)]
        color_map = {f"L{i+1}": colors[i] for i in range(args.levels)}
        bar_colors = [color_map[level] for level in region_err["level"]]
        ax.bar(region_err[args.region_col].astype(str), region_err["mean_abs_error"], color=bar_colors)
        # legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[f"L{i+1}"]) for i in range(args.levels)]
        labels = [f"L{i+1}" for i in range(args.levels)]
        ax.legend(handles, labels, title="Level", loc="best")
    else:
        ax.bar(region_err[args.region_col].astype(str), region_err["mean_abs_error"], color="#4c72b0")
    ax.set_title(f"SD Region Prediction Difficulty ({args.agg.upper()} MAE)")
    ax.set_xlabel(args.region_col)
    ax.set_ylabel("Mean Absolute Error")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    out_pdf = out_dir / args.out_pdf
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
