"""
Remap ASTGCN id_mean_abs_error.csv item_id from node index (0..N-1)
to sd_meta ID (sensor ID like 1114091) so it can be compared with Moirai2.
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--err-csv",
        required=True,
        help="Path to ASTGCN id_mean_abs_error.csv (item_id = node index)",
    )
    parser.add_argument(
        "--meta-csv",
        default=str(Path(__file__).resolve().parents[1] / "data" / "sd" / "sd_meta.csv"),
        help="Path to sd_meta.csv (row order = ASTGCN node order)",
    )
    parser.add_argument(
        "--out",
        help="Output path (default: <err-csv>_remapped.csv)",
    )
    args = parser.parse_args()

    err_path = Path(args.err_csv)
    meta_path = Path(args.meta_csv)
    out_path = Path(args.out) if args.out else err_path.parent / (err_path.stem + "_remapped.csv")

    df = pd.read_csv(err_path)
    meta = pd.read_csv(meta_path)

    # ASTGCN node index i = sd_meta row i's ID
    id_map = dict(zip(range(len(meta)), meta["ID"].astype(int)))
    df["item_id"] = df["item_id"].astype(int).map(id_map)

    df.to_csv(out_path, index=False)
    print(f"Remapped {len(df)} rows, saved to {out_path}")


if __name__ == "__main__":
    main()
