"""
Compare prediction errors between two models for each node.
Visualize the error difference on a map.
"""
from io import BytesIO
from math import log, pi, sin
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--err1-path",
        type=str,
        required=True,
        help="Path to first id_mean_abs_error.csv (baseline model)",
    )
    parser.add_argument(
        "--err2-path",
        type=str,
        required=True,
        help="Path to second id_mean_abs_error.csv (comparison model)",
    )
    parser.add_argument(
        "--model1-name",
        type=str,
        default="Model 1",
        help="Name of first model for legend",
    )
    parser.add_argument(
        "--model2-name",
        type=str,
        default="Model 2",
        help="Name of second model for legend",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "sd" / "sd_meta.csv"),
        help="Path to sd_meta.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default="model_comparison.pdf",
        help="Output PDF filename",
    )
    parser.add_argument(
        "--basemap-image",
        type=str,
        default="basemap_sd.png",
        help="Basemap image filename in output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    err1_path = Path(args.err1_path)
    err2_path = Path(args.err2_path)
    meta_path = Path(args.meta_path)
    out_dir = Path(args.out_dir)
    out_pdf = out_dir / args.out_pdf
    basemap_image = out_dir / args.basemap_image

    # Load error data
    err1_df = pd.read_csv(err1_path)
    err2_df = pd.read_csv(err2_path)
    meta_df = pd.read_csv(meta_path)

    # Ensure item_id is int
    err1_df["item_id"] = err1_df["item_id"].astype(int)
    err2_df["item_id"] = err2_df["item_id"].astype(int)

    # Merge the two error dataframes
    merged_err = err1_df.merge(
        err2_df,
        on="item_id",
        how="inner",
        suffixes=("_model1", "_model2"),
    )

    # Calculate error difference (model2 - model1)
    # Positive means model2 has higher error (worse)
    # Negative means model2 has lower error (better)
    merged_err["error_diff"] = merged_err["mean_abs_error_model2"] - merged_err["mean_abs_error_model1"]

    # Merge with metadata for location
    merged = meta_df.merge(merged_err, left_on="ID", right_on="item_id", how="inner")
    if merged.empty:
        raise ValueError("No matching IDs found between meta and error CSV.")

    print(f"Comparing {len(merged)} nodes")
    print(f"\n{args.model1_name} mean error: {merged['mean_abs_error_model1'].mean():.2f}")
    print(f"{args.model2_name} mean error: {merged['mean_abs_error_model2'].mean():.2f}")
    print(f"Mean error difference (Model2 - Model1): {merged['error_diff'].mean():.2f}")
    print(f"\nNodes where {args.model2_name} is better (lower error): {(merged['error_diff'] < 0).sum()}")
    print(f"Nodes where {args.model1_name} is better (lower error): {(merged['error_diff'] > 0).sum()}")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(17, 8))
    
    # Map visualization (left plot)
    ax1 = fig.add_subplot(121)
    
    min_lon, max_lon = merged["Lng"].min(), merged["Lng"].max()
    min_lat, max_lat = merged["Lat"].min(), merged["Lat"].max()
    pad_lon = (max_lon - min_lon) * 0.05
    pad_lat = (max_lat - min_lat) * 0.05
    min_lon -= pad_lon
    max_lon += pad_lon
    min_lat -= pad_lat
    max_lat += pad_lat
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    def _zoom_for_bounds(
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        width_px: int,
        height_px: int,
    ) -> int:
        def lat_rad(lat: float) -> float:
            s = sin(lat * pi / 180.0)
            s = min(max(s, -0.9999), 0.9999)
            return log((1 + s) / (1 - s)) / 2

        lat_fraction = (lat_rad(max_lat) - lat_rad(min_lat)) / pi
        lon_fraction = (max_lon - min_lon) / 360.0
        if lat_fraction <= 0 or lon_fraction <= 0:
            return 11
        zoom_lat = int(log(height_px / 256 / lat_fraction, 2))
        zoom_lon = int(log(width_px / 256 / lon_fraction, 2))
        return max(0, min(18, min(zoom_lat, zoom_lon)))

    try:
        if basemap_image.exists():
            img = Image.open(basemap_image)
        else:
            width_px, height_px = 1200, 900
            zoom = _zoom_for_bounds(min_lon, min_lat, max_lon, max_lat, width_px, height_px)
            url = (
                "https://staticmap.openstreetmap.de/staticmap.php?"
                f"center={center_lat},{center_lon}&zoom={zoom}&size={width_px}x{height_px}"
                "&maptype=mapnik"
            )
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
        ax1.imshow(
            img,
            extent=[min_lon, max_lon, min_lat, max_lat],
            origin="upper",
            zorder=0,
        )
    except Exception as exc:
        print(f"Basemap fetch failed: {exc}")

    # Use diverging colormap: blue = model2 better, red = model1 better
    max_abs_diff = max(abs(merged["error_diff"].min()), abs(merged["error_diff"].max()))
    sc1 = ax1.scatter(
        merged["Lng"],
        merged["Lat"],
        c=merged["error_diff"],
        cmap="RdBu_r",  # Red = positive (model2 worse), Blue = negative (model2 better)
        s=30,
        alpha=0.8,
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        zorder=1,
        edgecolors='black',
        linewidths=0.5,
    )

    ax1.set_title(f"Error Difference: {args.model2_name} - {args.model1_name}\n(Red={args.model1_name} better, Blue={args.model2_name} better)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_aspect("equal", adjustable="box")
    
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label("Error Difference (MAE)")

    # Scatter plot comparison (right plot)
    ax2 = fig.add_subplot(122)
    
    # Plot diagonal line (y=x)
    min_val = min(merged["mean_abs_error_model1"].min(), merged["mean_abs_error_model2"].min())
    max_val = max(merged["mean_abs_error_model1"].max(), merged["mean_abs_error_model2"].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1, label='Equal Error')
    
    # Color points by error difference
    sc2 = ax2.scatter(
        merged["mean_abs_error_model1"],
        merged["mean_abs_error_model2"],
        c=merged["error_diff"],
        cmap="RdBu_r",
        s=30,
        alpha=0.7,
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        edgecolors='black',
        linewidths=0.5,
    )
    
    ax2.set_xlabel(f"{args.model1_name} MAE", fontsize=12)
    ax2.set_ylabel(f"{args.model2_name} MAE", fontsize=12)
    ax2.set_title("Per-Node Error Comparison", fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation with statistics
    stats_text = f"Points below line: {args.model2_name} better\n"
    stats_text += f"Points above line: {args.model1_name} better\n"
    stats_text += f"Mean diff: {merged['error_diff'].mean():.2f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.7), fontsize=9)
    
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Error Difference")

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_pdf}")
    
    # Save comparison CSV
    csv_out = out_dir / (out_pdf.stem + "_comparison.csv")
    merged[['ID', 'Lat', 'Lng', 'mean_abs_error_model1', 'mean_abs_error_model2', 'error_diff']].to_csv(csv_out, index=False)
    print(f"Saved comparison CSV: {csv_out}")


if __name__ == "__main__":
    main()
