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

Generate dataset for Gift-eval. The generated files will be saved at `dataset/LargeST/gift_eval/[ca/gba/gla/sd]_[train_val]/[year]/[frequency]`
```bash
uv run ./scripts/data_processing/LargeST/generate_data_for_gift_eval.py --dataset sd --years 2019
```

Run benchmarks
```bash
cd benchmark
# ML
uv run --project ml python -m ml.ml_ensemble
uv run --project ml python -m ml.ml_methods

# TODO: move this somewhere
uv run --project feedforward python -m feedforward.feedforward
uv run --project feedforward python -m feedforward.naive

# FM
# Third-party repos used by FM entrypoints are expected under benchmark/fm/envs.
mkdir -p fm/envs

# FlowState dependency: granite-tsfm
mkdir -p fm/envs/flowstate
test -d fm/envs/flowstate/granite-tsfm || \
  git clone https://github.com/ibm-granite/granite-tsfm.git fm/envs/flowstate/granite-tsfm

# Kairos dependency: Kairos repository
mkdir -p fm/envs/kairos
test -d fm/envs/kairos/Kairos || \
  git clone --depth 1 https://github.com/foundation-model-research/Kairos.git fm/envs/kairos/Kairos

# TabPFN-TS dependency: tabpfn-time-series (pin to v1.0.0)
mkdir -p fm/envs/tabpfn_ts
if [ ! -d fm/envs/tabpfn_ts/tabpfn-time-series ]; then
  git clone https://github.com/PriorLabs/tabpfn-time-series.git fm/envs/tabpfn_ts/tabpfn-time-series
  (cd fm/envs/tabpfn_ts/tabpfn-time-series && git checkout v1.0.0)
fi

# Toto dependency: toto repository
mkdir -p fm/envs/toto
test -d fm/envs/toto/toto || \
  git clone https://github.com/DataDog/toto.git fm/envs/toto/toto

# Run FM benchmarks
# FM entrypoints are intended to be run from benchmark/ as modules via python -m.
# Running the script files directly is not supported.
uv run --project fm/flowstate python -m fm.flowstate.flowstate
uv run --project fm/kairos python -m fm.kairos.kairos
uv run --project fm/moirai python -m fm.moirai.chronos_bolt
uv run --project fm/moirai python -m fm.moirai.moirai_small
uv run --project fm/moirai python -m fm.moirai.moirai2
uv run --project fm/sundial python -m fm.sundial.sundial
uv run --project fm/tabpfn_ts python -m fm.tabpfn_ts.tabpfn_ts
uv run --project fm/timesfm python -m fm.timesfm.timesfm
uv run --project fm/toto python -m fm.toto.toto

# GNN
# TODO: add others
uv run --project gnn -m gnn.agcrn
```

## Interpret

## Finetune
