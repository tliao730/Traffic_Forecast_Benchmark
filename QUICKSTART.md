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
# ML
uv run --project ml python -m ml.ml_ensemble
uv run --project ml python -m ml.ml_methods

# TODO: move this somewhere
uv run --project feedforward python -m feedforward.feedforward
uv run --project feedforward python -m feedforward.naive

# FM
uv run --project fm/flowstate python -m fm.flowstate.flowstate

cd /u/dcao1/workspace/TrafficFM/benchmark
KAIROS_PATH="$PWD/fm/envs/kairos/Kairos"; \
test -d "$KAIROS_PATH" || git clone --depth 1 https://github.com/foundation-model-research/Kairos.git "$KAIROS_PATH"; \

uv run --project fm/kairos python -m fm.kairos.kairos


uv run --project fm/moirai python -m fm.moirai.chronos_1
uv run --project fm/moirai python -m fm.moirai.moirai
uv run --project fm/moirai python -m fm.moirai.moirai2
uv run --project fm/sundial python -m fm.sundial.sundial
uv run --project fm/tabpfn_ts python -m fm.tabpfn_ts.tabpfn_ts

mkdir -p benchmark/fm/envs/tabpfn_ts
cd benchmark/fm/envs/tabpfn_ts
git clone https://github.com/PriorLabs/tabpfn-time-series.git
cd tabpfn-time-series
git checkout v1.0.0

uv run --project fm/timesfm python -m fm.timesfm.timesfm

mkdir -p /work/nvme/bevu/dcao1/TrafficFM/benchmark/fm/envs/toto
cd /work/nvme/bevu/dcao1/TrafficFM/benchmark/fm/envs/toto
git clone https://github.com/DataDog/toto.git

uv run --project fm/toto python -m fm.toto.toto

# GNN
# TODO: add others
uv run --project gnn -m gnn.agcrn
```

## Interpret

## Finetune
