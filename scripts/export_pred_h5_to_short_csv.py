from pathlib import Path

import argparse
import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="Path to prediction h5 file")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--horizon", type=int, default=3, help="1-based horizon index")
    parser.add_argument("--ds_config", type=str, default="sd/2019/short")
    parser.add_argument("--chunk", type=int, default=200, help="Samples per chunk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = Path(args.h5)
    out_path = Path(args.out)
    horizon_idx = args.horizon - 1

    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if horizon_idx < 0:
        raise ValueError("--horizon must be >= 1")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        pred = f["pred"]
        true = f["true"]
        times = f.get("times")
        node_ids = f.get("node_ids")

        if horizon_idx >= pred.shape[1]:
            raise ValueError(f"horizon index {args.horizon} exceeds available horizon {pred.shape[1]}")

        if node_ids is None:
            node_ids_arr = np.arange(pred.shape[2], dtype=np.int64)
        else:
            node_ids_arr = node_ids[:]

        header_written = False
        num_samples = pred.shape[0]
        for start in range(0, num_samples, args.chunk):
            end = min(num_samples, start + args.chunk)

            pred_chunk = pred[start:end, horizon_idx, :]
            true_chunk = true[start:end, horizon_idx, :]

            if times is not None:
                time_chunk = times[start:end, horizon_idx]
                time_str = pd.to_datetime(time_chunk, unit="ns", utc=True).tz_convert(None)
                time_str = time_str.strftime("%Y-%m-%d %H:%M")
            else:
                time_str = np.array([""] * (end - start))

            sample_idx = np.arange(start, end, dtype=np.int64)

            # Build rows: (sample, node) for truth and mean
            # Repeat per node
            sample_rep = np.repeat(sample_idx, len(node_ids_arr))
            node_rep = np.tile(node_ids_arr, end - start)
            time_rep = np.repeat(time_str, len(node_ids_arr))

            # truth
            truth_vals = true_chunk.reshape(-1)
            df_truth = pd.DataFrame(
                {
                    "ds_config": args.ds_config,
                    "sample_idx": sample_rep,
                    "item_id": node_rep,
                    "timestamp": time_rep,
                    "dim": 0,
                    "stat": "truth",
                    "value": truth_vals,
                }
            )

            # pred
            pred_vals = pred_chunk.reshape(-1)
            df_pred = pd.DataFrame(
                {
                    "ds_config": args.ds_config,
                    "sample_idx": sample_rep,
                    "item_id": node_rep,
                    "timestamp": time_rep,
                    "dim": 0,
                    "stat": "mean",
                    "value": pred_vals,
                }
            )

            out_df = pd.concat([df_truth, df_pred], ignore_index=True)
            out_df.to_csv(out_path, index=False, mode="a", header=not header_written)
            header_written = True


if __name__ == "__main__":
    main()
