from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sd" / "sd_meta.csv"
ERR_PATH = Path(__file__).resolve().parents[0] / "id_mean_abs_error.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_HTML = OUT_DIR / "sd_meta_map_5levels.html"

LEVEL_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    err_df = pd.read_csv(ERR_PATH)
    err_df["item_id"] = err_df["item_id"].astype(int)
    err_df["level"] = pd.qcut(err_df["mean_abs_error"], 5, labels=False, duplicates="drop")
    df = df.merge(err_df, left_on="ID", right_on="item_id", how="inner")

    if df.empty:
        raise ValueError("No matching IDs found in sd_meta.csv")

    center_lat = df["Lat"].mean()
    center_lng = df["Lng"].mean()

    m = folium.Map(location=[center_lat, center_lng], zoom_start=10, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        level = int(row["level"]) if pd.notna(row["level"]) else 0
        color = LEVEL_COLORS[level]
        popup = (
            f"ID: {row['ID']}<br>Fwy: {row['Fwy']}<br>Dir: {row['Direction']}"
            f"<br>MAE: {row['mean_abs_error']:.3f}<br>Level: {level + 1}"
        )
        folium.CircleMarker(
            location=[row["Lat"], row["Lng"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.8,
            fill_color=color,
            popup=popup,
        ).add_to(cluster)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
