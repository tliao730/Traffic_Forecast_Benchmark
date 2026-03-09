import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value

TARGET_DIR = Path("dataset/LargeST/gift_eval")


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def build_time_features(index, add_time_of_day, add_day_of_week):
    features = []
    if add_time_of_day:
        # Fraction of the day elapsed (0-1 range)
        frac_day = ((index - index.normalize()) / pd.Timedelta(days=1)).astype(float)
        features.append(frac_day.astype(np.float32))
    if add_day_of_week:
        dow = (index.dayofweek / 7).astype(float)
        features.append(dow.astype(np.float32))

    if features:
        return np.asarray(features, dtype=np.float32)
    return None


def prepare_output_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{path}' already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)


def build_records(test_df, freq, time_features):
    start_timestamp = test_df.index[0].isoformat()
    dynamic_template = None
    if time_features is not None:
        dynamic_template = [feat.tolist() for feat in time_features]

    records = []
    for column in test_df.columns:
        record = {
            "item_id": str(column),
            "start": start_timestamp,
            "freq": freq,
            "target": test_df[column].astype(np.float32).tolist(),
        }
        if dynamic_template is not None:
            record["past_feat_dynamic_real"] = [feat[:] for feat in dynamic_template]
        records.append(record)
    return records


def save_as_arrow(records, output_path, has_dynamic_features):
    features = Features(
        {
            "item_id": Value("string"),
            "start": Value("string"),
            "freq": Value("string"),
            "target": Sequence(Value("float32")),
        }
    )
    if has_dynamic_features:
        features["past_feat_dynamic_real"] = Sequence(
            feature=Sequence(Value("float32"))
        )

    dataset = Dataset.from_list(records, features=features)
    dataset.save_to_disk(output_path)
    return dataset


def generate_train_val_test(args):
    years = args.years.split("_")
    df = pd.DataFrame()
    for y in years:
        # Prefer processed file name, fallback to raw if needed.
        data_path = Path("data/LargeST") / args.dataset / f"{args.dataset}_his_{y}.h5"
        df_tmp = pd.read_hdf(data_path)
        df = pd.concat([df, df_tmp])

    print("original data shape:", df.shape)
    df = df.resample(args.freq).mean().round(0)
    df = df.sort_index()
    num_samples = len(df)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)

    train = df.iloc[:num_train]
    val = df.iloc[num_train : num_train + num_val]
    test = df.iloc[num_train + num_val :]

    if test.empty:
        raise ValueError(
            "Test split is empty. Check the provided years or split ratios."
        )

    def _save_split(split_name: str, split_df: pd.DataFrame, output_path: Path):
        if split_df.empty:
            raise ValueError(
                f"{split_name} split is empty. Check years/freq/split ratios."
            )

        time_features = build_time_features(
            split_df.index, bool(args.tod), bool(args.dow)
        )
        prepare_output_dir(output_path, args.overwrite)
        records = build_records(split_df, args.freq, time_features)
        dataset = save_as_arrow(records, output_path, time_features is not None)
        print(
            f"Saved {split_name} HuggingFace dataset with {dataset.num_rows} series to '{output_path}'."
        )

    # NOTE:
    # - Keep the existing (legacy) output path for test unchanged.
    # - Save train/val as separate datasets alongside it.
    # output_test = Path(args.output_dir) / args.dataset / args.years / args.freq
    output_train = TARGET_DIR / f"{args.dataset}_train" / args.years / args.freq
    output_val = TARGET_DIR / f"{args.dataset}_val" / args.years / args.freq
    print("train/val/test shapes:", train.shape, val.shape, test.shape)
    print("test start index in original data:", num_train + num_val)
    print("test start timestamp:", test.index[0])

    # _save_split('test', test, output_test)
    _save_split("train", train, output_train)
    _save_split("val", val, output_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ca", help="dataset name")
    parser.add_argument(
        "--years",
        type=str,
        default="2019",
        help="if use data from multiple years, please use underline to separate them, e.g., 2018_2019",
    )
    parser.add_argument("--tod", type=int, default=1, help="time of day")
    parser.add_argument("--dow", type=int, default=1, help="day of week")
    parser.add_argument(
        "--freq",
        type=str,
        default="15T",
        help="sampling frequency, aligned with pandas offsets (e.g., 15T)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing arrow dataset directory if it exists",
    )

    args = parser.parse_args()
    generate_train_val_test(args)
