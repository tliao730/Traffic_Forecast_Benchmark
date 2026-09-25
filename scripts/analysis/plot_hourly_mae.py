#!/usr/bin/env python3
"""Hour-of-day MAE from prediction dumps, for any family.

Both prediction writers emit the same schema
(``ds_config, sample_idx, item_id, timestamp, dim, stat, value``): the FM /
SSM / linear path through ``common.save_predictions_csv`` and the GNN path
through ``BaseEngine._save_predictions_csv``. So one script covers both.

One model per figure. Outputs are filed by region and year, taken from the
CSV's own ``ds_config`` rather than from the filename:

    <out-root>/<region>/<year>/<label>_<term>.pdf
    <out-root>/<region>/<year>/<label>_<term>.csv   # the plotted numbers

    python scripts/analysis/plot_hourly_mae.py \
        --csv .../astgcn_sd_2018_long_predictions.csv --labels ASTGCN

    # several models: still one figure each, filed in the same tree
    python scripts/analysis/plot_hourly_mae.py \
        --csv a.csv --csv b.csv --labels ASTGCN,Moirai2

``--out`` overrides the path for a single CSV. ``--per-step`` scores one
forecast step (1-based) instead of averaging all steps in the window, matching
the GNN "Horizon k" convention.

The hour is the hour of the *forecast target* timestamp, not of the context,
which is what makes the curve readable as "how hard is this time of day".
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_OUT_ROOT = "img/hourly_mae"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", action="append", required=True,
                   help="prediction CSV; repeat for several models")
    p.add_argument("--labels", type=str, default="",
                   help="comma-separated model names (default: file stems)")
    p.add_argument("--label", type=str, default="",
                   help="single-model convenience alias for --labels")
    p.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT,
                   help=f"root of the region/year tree (default: {DEFAULT_OUT_ROOT})")
    p.add_argument("--out", type=str, default="",
                   help="explicit output path; only valid with a single --csv")
    p.add_argument("--ext", type=str, default="pdf", choices=["pdf", "png"],
                   help="figure format when using --out-root (default: pdf)")
    p.add_argument("--tag", type=str, default="",
                   help="suffix for the output stem, e.g. prof250, to keep a "
                        "profiling figure from overwriting the benchmark one")
    p.add_argument("--per-step", type=int, default=0,
                   help="score only this forecast step (1-based); 0 = all steps")
    p.add_argument("--title", type=str, default="",
                   help="figure title (default: '<label> -- <region>/<year>/<term>')")
    return p.parse_args()


def hourly_mae(csv_path, per_step=0):
    """Return a 24-row frame (hour, mean_abs_error, n) for one prediction CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    keys = ["ds_config", "sample_idx", "item_id", "timestamp", "dim"]
    truth = df[df["stat"] == "truth"]
    pred_stat = "mean" if "mean" in set(df["stat"]) else df["stat"].iloc[0]
    pred = df[df["stat"] == pred_stat]
    merged = truth.merge(pred, on=keys, how="inner", suffixes=("_truth", "_pred"))
    if merged.empty:
        raise SystemExit(f"{csv_path}: no truth/{pred_stat} pairs matched")

    if per_step:
        # Step index within each window: rank of the timestamp inside a
        # (sample_idx, item_id) group. Windows are non-overlapping, so this is
        # exactly the forecast horizon step.
        merged["step"] = (
            merged.groupby(["sample_idx", "item_id"])["timestamp"].rank(method="first")
        )
        merged = merged[merged["step"] == per_step]
        if merged.empty:
            raise SystemExit(f"{csv_path}: no rows at step {per_step}")

    merged["abs_error"] = (merged["value_pred"] - merged["value_truth"]).abs()
    merged["hour"] = merged["timestamp"].dt.hour
    out = (merged.groupby("hour")
                 .agg(mean_abs_error=("abs_error", "mean"), n=("abs_error", "size"))
                 .reindex(range(24))
                 .reset_index())
    out.attrs["ds_config"] = str(merged["ds_config"].iloc[0])
    return out


def split_ds_config(ds_config):
    """'sd/2018/long' -> ('sd', '2018', 'long'); tolerate anything else."""
    parts = ds_config.split("/")
    while len(parts) < 3:
        parts.append("unknown")
    return parts[0], parts[1], parts[2]


def plot_one(frame, label, fig_path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(frame["hour"], frame["mean_abs_error"], width=0.8,
           color="#5B9BD5", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Mean Absolute Error", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.grid(True, alpha=0.3, axis="y")
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    labels = [s for s in (args.labels or args.label).split(",") if s]
    if not labels:
        labels = [os.path.basename(c).replace("_predictions.csv.gz", "")
                             .replace("_predictions.csv", "") for c in args.csv]
    if len(labels) != len(args.csv):
        raise SystemExit(f"{len(labels)} labels for {len(args.csv)} CSVs")
    if args.out and len(args.csv) > 1:
        raise SystemExit("--out takes a single --csv; use --out-root for several")

    for label, csv_path in zip(labels, args.csv):
        frame = hourly_mae(csv_path, args.per_step)
        region, year, term = split_ds_config(frame.attrs["ds_config"])
        step_note = f", step {args.per_step}" if args.per_step else ""
        title = args.title or f"{label} — {region}/{year}/{term}{step_note}"

        stem = f"{label}_{term}"
        if args.tag:
            stem += f"_{args.tag}"
        if args.per_step:
            stem += f"_h{args.per_step}"
        if args.out:
            fig_path = args.out
            num_path = os.path.splitext(args.out)[0] + ".csv"
        else:
            out_dir = os.path.join(args.out_root, region, year)
            fig_path = os.path.join(out_dir, f"{stem}.{args.ext}")
            num_path = os.path.join(out_dir, f"{stem}.csv")

        plot_one(frame, label, fig_path, title)

        numbers = frame.assign(model=label, ds_config=frame.attrs["ds_config"],
                               step=args.per_step or "all")
        numbers.to_csv(num_path, index=False)

        worst = frame.loc[frame["mean_abs_error"].idxmax()]
        best = frame.loc[frame["mean_abs_error"].idxmin()]
        print(f"{label}: mean {frame['mean_abs_error'].mean():.2f} | "
              f"worst {int(worst['hour']):02d}h {worst['mean_abs_error']:.2f} | "
              f"best {int(best['hour']):02d}h {best['mean_abs_error']:.2f} "
              f"-> {fig_path}")


if __name__ == "__main__":
    main()
