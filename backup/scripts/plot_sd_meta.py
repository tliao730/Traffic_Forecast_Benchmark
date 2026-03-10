from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sd" / "sd_meta.csv"
ADJ_PATH = Path(__file__).resolve().parents[1] / "data" / "sd" / "sd_rn_adj.npy"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_HTML = OUT_DIR / "sd_meta_map.html"
EDGE_THRESHOLD = 0.001
MAX_EDGES = 6000


def main() -> None:
    df = pd.read_csv(CSV_PATH)
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

    if ADJ_PATH.exists():
        adj = np.load(ADJ_PATH)
        edge_group = folium.FeatureGroup(name="RoadNetwork", show=True)
        edge_count = 0
        for i in range(adj.shape[0]):
            for j in range(i + 1, adj.shape[1]):
                w = adj[i, j]
                if w >= EDGE_THRESHOLD:
                    folium.PolyLine(
                        locations=[
                            [df.iloc[i]["Lat"], df.iloc[i]["Lng"]],
                            [df.iloc[j]["Lat"], df.iloc[j]["Lng"]],
                        ],
                        color="#ff7f0e",
                        weight=1,
                        opacity=0.4,
                    ).add_to(edge_group)
                    edge_count += 1
                    if edge_count >= MAX_EDGES:
                        break
            if edge_count >= MAX_EDGES:
                break
        edge_group.add_to(m)
        folium.LayerControl().add_to(m)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
