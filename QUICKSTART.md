# Traffic Forecasting Benchmarking Platform

## Quick Start

Clone the repo
```bash
git clone https://github.com/DC-research/TrafficFM.git
cd TrafficFM
```

Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install base dependencies
```bash
uv sync
```

Go to Kaggle -> Settings -> API, create a kaggle API token, then add it to `~/.kaggle/kaggle.json`
```json
{
  "username": "USERNAME",
  "key": "KAGGLE_API_KEY"
}
```
Change the ownership
```bash
chmod 600 ~/.kaggle/kaggle.json
```

Download `LargeST` dataset, this raw dataset should be saved to `data/LargeSt/raw`
```bash
./scripts/data_processing/LargeST/download_dataset.sh
```

Process data based on regions, the processed data will be saved at `data/LargeST/[ca/gba/gla/sd]`
```bash
uv run ./scripts/data_processing/LargeST/process_dataset.py --year 2019
```

Generate dataset for GNN training, the dataset will be saved at `dataset/LargeST/baseline/[ca/gba/gla/sd]/[year]`
```bash
uv run ./scripts/data_processing/LargeST/generate_dataset_for_training.py --dataset sd --years 2019
```

Generate dataset for Gift-eval, the dataset will be saved at `dataset/LargeST/gift_eval/[ca/gba/gla/sd]_[train_val]/[year]/[freqency]`
```bash
uv run ./scripts/data_processing/LargeST/generate_data_for_gift_eval.py --dataset sd --years 2019
```

Run benchmark
```bash
cd benchmark
uv run --project arima python -m arima.arima # run as module
uv run --project feedforward python -m feedforward.feedforward
uv run --project feedforward python -m feedforward.naive
uv run --project flowstate python -m flowstate.flowstate
uv run --project kairos python -m kairos.kairos
uv run --project moirai python -m moirai.chronos_1
uv run --project moirai python -m moirai.moirai
uv run --project moirai python -m moirai.moirai2
uv run --project sundial python -m sundial.sundial
uv run --project tabpfn_ts python -m tabpfn_ts.ml_ensemble
uv run --project tabpfn_ts python -m tabpfn_ts.ml_methods
uv run --project tabpfn_ts python -m tabpfn_ts.tabpfn_ts
uv run --project timesfm_eval python -m timesfm_eval.timesfm_eval
uv run --project toto python -m toto.toto
```
