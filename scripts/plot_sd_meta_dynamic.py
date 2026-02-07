from pathlib import Path

import folium
import h5py
import numpy as np
import pandas as pd
from folium.plugins import TimestampedGeoJson

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sd" / "sd_meta.csv"
H5_PATH = Path(__file__).resolve().parents[1] / "data" / "sd" / "sd_his_raw_2019.h5"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_HTML = OUT_DIR / "sd_meta_map_dynamic.html"

# Sampling controls (reduce file size / rendering cost)
TIME_STRIDE = 12  # 5-min data -> 12 steps = 1 hour
STATION_STRIDE = 1
MAX_TIMESTEPS = 720  # limit frames (720 hours ~= 30 days)

RADIUS = 4
OPACITY = 0.85


def value_to_color(val: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(val):
        return "#999999"
    if vmax == vmin:
        ratio = 0.5
    else:
        ratio = (val - vmin) / (vmax - vmin)
    ratio = min(max(ratio, 0.0), 1.0)
    r = int(255 * ratio)
    g = int(80 * (1 - ratio))
    b = int(255 * (1 - ratio))
    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    center_lat = df["Lat"].mean()
    center_lng = df["Lng"].mean()

    m = folium.Map(location=[center_lat, center_lng], zoom_start=10, tiles="OpenStreetMap")

    if not H5_PATH.exists():
        raise FileNotFoundError(f"Missing H5 file: {H5_PATH}")

    with h5py.File(H5_PATH, "r") as f:
        ids = [x.decode("utf-8") for x in f["t/axis0"][:]]
        times_ns = f["t/axis1"][:]
        values = f["t/block0_values"]

        df_indexed = df.set_index("ID")
        df_indexed = df_indexed.loc[[int(i) for i in ids]]

        lat = df_indexed["Lat"].values
        lng = df_indexed["Lng"].values

        time_idx = np.arange(0, len(times_ns), TIME_STRIDE)[:MAX_TIMESTEPS]
        station_idx = np.arange(0, len(ids), STATION_STRIDE)

        time_strings = pd.to_datetime(times_ns[time_idx], unit="ns", utc=True).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        sample_values = values[time_idx][:, station_idx]
        finite_vals = sample_values[np.isfinite(sample_values)]
        if finite_vals.size == 0:
            raise ValueError("No finite values found in sampled data.")
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))

        features = []
        for t_pos, t in enumerate(time_idx):
            ts = time_strings[t_pos]
            row = values[t][station_idx]
            for s_pos, s in enumerate(station_idx):
                val = float(row[s_pos])
                color = value_to_color(val, vmin, vmax)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(lng[s]), float(lat[s])],
                        },
                        "properties": {
                            "time": ts,
                            "style": {
                                "color": color,
                                "fillColor": color,
                                "fillOpacity": OPACITY,
                                "weight": 1,
                                "radius": RADIUS,
                            },
                            "icon": "circle",
                            "popup": f"ID: {ids[s]}<br>Time: {ts}<br>Value: {val:.3f}",
                        },
                    }
                )

        TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            period="PT5M",
            add_last_point=False,
            auto_play=True,
            loop=True,
            max_speed=3,
            loop_button=True,
            date_options="YYYY-MM-DD HH:mm",
            time_slider_drag_update=True,
        ).add_to(m)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
