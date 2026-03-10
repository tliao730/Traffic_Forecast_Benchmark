from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sd" / "sd_meta.csv"
IDS_PATH = Path(__file__).resolve().parents[0] / "top_20_hardest_ids.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_HTML = OUT_DIR / "sd_meta_map_top20.html"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    ids_df = pd.read_csv(IDS_PATH)
    ids = ids_df["item_id"].astype(int).tolist()
    df = df[df["ID"].isin(ids)].copy()

    if df.empty:
        raise ValueError("No matching IDs found in sd_meta.csv")

    center_lat = df["Lat"].mean()
    center_lng = df["Lng"].mean()

    m = folium.Map(location=[center_lat, center_lng], zoom_start=10, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        popup = f"ID: {row['ID']}<br>Fwy: {row['Fwy']}<br>Dir: {row['Direction']}"
        folium.CircleMarker(
            location=[row["Lat"], row["Lng"]],
            radius=3,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(cluster)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
