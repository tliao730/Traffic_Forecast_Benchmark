from io import BytesIO
from math import log, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "gba" / "gba_meta.csv"
ERR_PATH = Path(__file__).resolve().parents[0] / "gba" / "id_mean_abs_error.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_PDF = OUT_DIR / "gba_id_error.pdf"
BASEMAP_IMAGE = OUT_DIR / "basemap_gba.png"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    err_df = pd.read_csv(ERR_PATH)
    err_df["item_id"] = err_df["item_id"].astype(int)

    merged = df.merge(err_df, left_on="ID", right_on="item_id", how="inner")
    if merged.empty:
        raise ValueError("No matching IDs found between meta and error CSV.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8.5))
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
        if BASEMAP_IMAGE.exists():
            img = Image.open(BASEMAP_IMAGE)
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
        ax.imshow(
            img,
            extent=[min_lon, max_lon, min_lat, max_lat],
            origin="upper",
            zorder=0,
        )
    except Exception as exc:
        print(f"Basemap fetch failed: {exc}")

    sc = ax.scatter(
        merged["Lng"],
        merged["Lat"],
        c=merged["mean_abs_error"],
        cmap="viridis",
        s=18,
        alpha=0.9,
        zorder=1,
    )

    for _, row in merged.iterrows():
        ax.text(
            row["Lng"],
            row["Lat"],
            f"{row['mean_abs_error']:.1f}",
            fontsize=3,
            ha="center",
            va="center",
            color="black",
        )

    ax.set_title("GBA Nodes Prediction Difficulty (MAE)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean Absolute Error")

    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
