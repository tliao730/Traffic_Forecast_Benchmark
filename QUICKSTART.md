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

Download `LargeST` dataset, this raw dataset should be saved to `data/LargeSt/raw`
```bash
./scripts/data/download_dataset.sh
```

Process data based on regions, the processed data will be saved at `data/LargeST/[ca/gba/gla/sd]`
```bash
uv run ./scripts/data/LargeST/process_dataset.py --year 2019
```

Generate dataset for GNN training, the dataset will be saved at `dataset/LargeST/baseline/[ca/gba/gla/sd]/[year]`
```bash
uv run ./scripts/data/LargeST/generate_dataset_for_training.py --dataset sd --years 2019
```

Generate dataset for Gift-eval, the dataset will be saved at `dataset/LargeST/gift_eval/[ca/gba/gla/sd]_[train_eval]/[year]/[freqency]`
```bash
uv run ./scripts/data/LargeST/generate_data_for_gift_eval.py --dataset sd --years 2019
```

Run benchmark
```bash
uv run --project benchmark/arima benchmark/arima/arima.py 
```