import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_DIR = Path("data/LargeST/raw")
DISTRICTS = {
    "gba": [4],
    "gla": [7, 8, 12],
    "sd": [11],
}


def main(year):
    target_dir = Path("data/LargeST") / "ca"
    h5_path = target_dir / f"ca_his_{year}.h5"
    if h5_path.exists():
        print("CA historical data already exists, skipping...")
    else:
        print("Processing CA historical data...")
        target_dir.mkdir(exist_ok=True)
        ca_his = pd.read_hdf(RAW_DATA_DIR / f"ca_his_raw_{year}.h5")
        ca_his = ca_his.resample("15T").mean().round(0)
        ca_his = ca_his.fillna(0)
        ca_his.to_hdf(h5_path, key="t", mode="w")
        shutil.copy(RAW_DATA_DIR / "ca_meta.csv", target_dir / "ca_meta.csv")
        shutil.copy(RAW_DATA_DIR / "ca_rn_adj.npy", target_dir / "ca_rn_adj.npy")

    ca_meta = pd.read_csv(RAW_DATA_DIR / "ca_meta.csv")
    ca_rn_adj = np.load(RAW_DATA_DIR / "ca_rn_adj.npy")
    for region in ["gba", "gla", "sd"]:
        target_dir = Path("data/LargeST") / region
        h5_path = target_dir / f"{region}_his_{year}.h5"
        if h5_path.exists():
            print(f"{region.upper()} data already exists, skipping...")
            continue

        print(f"Processing {region.upper()} data...")
        target_dir.mkdir(exist_ok=True)
        region_meta = ca_meta[ca_meta.District.isin(DISTRICTS[region])]
        region_meta = region_meta.reset_index()
        region_meta = region_meta.drop(columns=["index"])
        region_meta.to_csv(target_dir / f"{region}_meta.csv", index=False)

        region_meta_id2 = region_meta.ID2.values.tolist()
        region_rn_adj = ca_rn_adj[region_meta_id2]
        region_rn_adj = region_rn_adj[:, region_meta_id2]
        np.save(target_dir / f"{region}_rn_adj.npy", region_rn_adj)

        region_meta.ID = region_meta.ID.astype(str)
        region_meta_id = region_meta.ID.values.tolist()

        ca_his = pd.read_hdf(target_dir / f"../ca/ca_his_{year}.h5")
        region_his = ca_his[region_meta_id]
        region_his.to_hdf(h5_path, key="t", mode="w")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year", type=int, default=2019, help="Year of the data to process"
    )
    args = parser.parse_args()
    main(args.year)
