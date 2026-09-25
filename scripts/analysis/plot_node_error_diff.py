#!/usr/bin/env python3
"""Per-sensor MAE comparison between two models: map + scatter.

Left panel: every sensor at its own lat/lon, coloured by MAE(A) - MAE(B), so
red marks where B wins and blue where A wins. Right panel: the same sensors as
MAE(A) against MAE(B) with the y=x line, which shows whether the difference is
a uniform shift or concentrated in the hard sensors.

Both prediction writers share one schema, so either family can be either side:
the GNN dump keys sensors by node index 0..N-1 and the FM/SSM/linear dump by
the real Caltrans ID. LargeST orders the his.npz columns, the gift_eval series
and {region}_meta.csv identically -- verified for sd and gba -- so the index is
just the row number in the meta file, which is how the two are joined here.

    python scripts/analysis/plot_node_error_diff.py \
        --csv-a .../astgcn_sd_2018_long_prof500_predictions.csv.gz --label-a ASTGCN \
        --csv-b .../sd_2018_long_predictions.csv.gz --label-b LRU --tag prof500

Outputs <out-root>/<region>/<year>/<A>_vs_<B>_<term>[_<tag>].pdf and a CSV of
the per-sensor numbers next to it.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_OUT_ROOT = "img/node_error"
META_TEMPLATE = "data/LargeST/{region}/{region}_meta.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-a", required=True, help="prediction dump for model A")
    p.add_argument("--csv-b", required=True, help="prediction dump for model B")
    p.add_argument("--label-a", default="A", help="display name for model A")
    p.add_argument("--label-b", default="B", help="display name for model B")
    p.add_argument("--meta", default="", help=f"sensor metadata CSV (default: {META_TEMPLATE})")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--out", default="", help="explicit output path, overriding --out-root")
    p.add_argument("--ext", default="pdf", choices=["pdf", "png"])
    p.add_argument("--tag", default="", help="suffix for the output stem, e.g. prof500")
    p.add_argument("--per-step", type=int, default=0,
                   help="score only this forecast step (1-based); 0 = all steps")
    return p.parse_args()


def node_mae(csv_path, per_step=0):
    """Per-sensor MAE for one dump. Returns (frame[item_id, mae, n], ds_config)."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    keys = ["ds_config", "sample_idx", "item_id", "timestamp", "dim"]
    truth = df[df["stat"] == "truth"]
    pred_stat = "mean" if "mean" in set(df["stat"]) else df["stat"].iloc[0]
    merged = truth.merge(df[df["stat"] == pred_stat], on=keys, how="inner",
                         suffixes=("_truth", "_pred"))
    if merged.empty:
        raise SystemExit(f"{csv_path}: no truth/{pred_stat} pairs matched")

    if per_step:
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], format="mixed",
                                             errors="coerce")
        merged["step"] = (
            merged.groupby(["sample_idx", "item_id"])["timestamp"].rank(method="first")
        )
        merged = merged[merged["step"] == per_step]

    merged["abs_error"] = (merged["value_pred"] - merged["value_truth"]).abs()
    out = (merged.groupby("item_id")
                 .agg(mae=("abs_error", "mean"), n=("abs_error", "size"))
                 .reset_index())
    return out, str(merged["ds_config"].iloc[0])


def to_sensor_id(frame, meta_ids):
    """
    Normalise item_id to the real sensor ID.

    A GNN dump numbers sensors 0..N-1; an FM/SSM dump carries the Caltrans ID,
    which is five or more digits. Anything that fits inside the sensor count is
    therefore an index, not an ID.
    """
    ids = pd.to_numeric(frame["item_id"], errors="coerce")
    if ids.isna().any():
        frame["sensor_id"] = frame["item_id"].astype(str)
        return frame, False
    if int(ids.max()) < len(meta_ids):
        frame["sensor_id"] = [str(meta_ids[int(i)]) for i in ids]
        return frame, True
    frame["sensor_id"] = ids.astype("int64").astype(str)
    return frame, False


def main():
    args = parse_args()

    a, cfg_a = node_mae(args.csv_a, args.per_step)
    b, cfg_b = node_mae(args.csv_b, args.per_step)
    if cfg_a != cfg_b:
        raise SystemExit(f"different cells: {cfg_a!r} vs {cfg_b!r} -- refusing to "
                         "compare models scored on different windows")
    region, year, term = (cfg_a.split("/") + ["unknown"] * 3)[:3]

    meta_path = args.meta or META_TEMPLATE.format(region=region)
    meta = pd.read_csv(meta_path)
    meta["sensor_id"] = meta["ID"].astype(str)
    meta_ids = meta["ID"].tolist()

    a, remapped_a = to_sensor_id(a, meta_ids)
    b, remapped_b = to_sensor_id(b, meta_ids)
    print(f"{args.label_a}: {len(a)} sensors"
          f"{' (node index remapped via meta order)' if remapped_a else ''}")
    print(f"{args.label_b}: {len(b)} sensors"
          f"{' (node index remapped via meta order)' if remapped_b else ''}")

    df = (a[["sensor_id", "mae"]].rename(columns={"mae": "mae_a"})
            .merge(b[["sensor_id", "mae"]].rename(columns={"mae": "mae_b"}),
                   on="sensor_id", how="inner")
            .merge(meta[["sensor_id", "Lat", "Lng"]], on="sensor_id", how="left"))
    if df.empty:
        raise SystemExit("no sensors in common after the ID join")
    missing = int(df["Lat"].isna().sum())
    if missing:
        print(f"warning: {missing} sensors have no coordinates; dropped from the map")
    df["diff"] = df["mae_a"] - df["mae_b"]

    lim = float(np.nanpercentile(df["diff"].abs(), 99)) or 1.0
    fig, (ax_map, ax_sc) = plt.subplots(1, 2, figsize=(15, 6.5))

    geo = df.dropna(subset=["Lat", "Lng"])
    sc = ax_map.scatter(geo["Lng"], geo["Lat"], c=geo["diff"], cmap="RdBu_r",
                        vmin=-lim, vmax=lim, s=26, edgecolors="black", linewidths=0.4)
    ax_map.set_title(f"Error Difference: {args.label_a} - {args.label_b}\n"
                     f"(Red = {args.label_b} better, Blue = {args.label_a} better)")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax_map, label="Error Difference (MAE)")

    sc2 = ax_sc.scatter(df["mae_b"], df["mae_a"], c=df["diff"], cmap="RdBu_r",
                        vmin=-lim, vmax=lim, s=26, edgecolors="black", linewidths=0.4)
    hi = float(max(df["mae_a"].max(), df["mae_b"].max())) * 1.05
    ax_sc.plot([0, hi], [0, hi], "--", color="grey", linewidth=0.9, label="Equal error")
    ax_sc.set_xlim(0, hi)
    ax_sc.set_ylim(0, hi)
    ax_sc.set_aspect("equal")
    ax_sc.set_xlabel(f"{args.label_b} MAE")
    ax_sc.set_ylabel(f"{args.label_a} MAE")
    ax_sc.set_title("Per-Node Error Comparison")
    ax_sc.grid(True, alpha=0.25)
    wins_b = int((df["diff"] > 0).sum())
    ax_sc.legend(loc="upper left", fontsize=9)
    ax_sc.text(0.03, 0.80,
               f"above line: {args.label_b} better ({wins_b}/{len(df)})\n"
               f"mean diff: {df['diff'].mean():+.2f}\n"
               f"median diff: {df['diff'].median():+.2f}",
               transform=ax_sc.transAxes, fontsize=9, va="top",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    fig.colorbar(sc2, ax=ax_sc, label="Error Difference")

    stem = f"{args.label_a}_vs_{args.label_b}_{term}"
    if args.tag:
        stem += f"_{args.tag}"
    if args.per_step:
        stem += f"_h{args.per_step}"
    if args.out:
        fig_path, num_path = args.out, os.path.splitext(args.out)[0] + ".csv"
    else:
        out_dir = os.path.join(args.out_root, region, year)
        fig_path = os.path.join(out_dir, f"{stem}.{args.ext}")
        num_path = os.path.join(out_dir, f"{stem}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)

    fig.suptitle(f"{cfg_a}" + (f", step {args.per_step}" if args.per_step else ""),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    df.assign(model_a=args.label_a, model_b=args.label_b, ds_config=cfg_a).to_csv(
        num_path, index=False)
    print(f"Wrote {fig_path}\nWrote {num_path}")
    print(f"{args.label_a} mean MAE {df['mae_a'].mean():.2f} | "
          f"{args.label_b} mean MAE {df['mae_b'].mean():.2f} | "
          f"{args.label_b} better on {wins_b}/{len(df)} sensors")


if __name__ == "__main__":
    main()
