#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   bash data_analysis/run_gift_eval_2019.sh
#
# This script runs the gift-eval exporter for 2019 on:
#   ca, sd, gka, gba
#
# Notes:
# - It runs from the `data/` directory so `--dataset ca` resolves to `data/ca/...`.
# - By default it uses --overwrite for train/val outputs.
# - Test output is NOT touched if you commented out test saving in the python script.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: data directory not found: ${DATA_DIR}" >&2
  exit 1
fi

cd "${DATA_DIR}"

PY_SCRIPT="${ROOT_DIR}/data_analysis/generate_data_for_gift_eval.py"
OUT_DIR="./data/gift_eval_datasets"
YEAR="2019"
FREQ="15T"

DATASETS=(ca sd gla gba)

for ds in "${DATASETS[@]}"; do
  run_ds="${ds}"


  if [[ ! -d "${run_ds}" ]]; then
    echo "WARN: dataset directory not found under data/: '${run_ds}' (skipping)"
    continue
  fi

  echo "=== Running dataset=${run_ds}, years=${YEAR}, freq=${FREQ} ==="
  python "${PY_SCRIPT}" \
    --dataset "${run_ds}" \
    --years "${YEAR}" \
    --freq "${FREQ}" \
    --output_dir "${OUT_DIR}" \
    --overwrite
done

echo "Done. Outputs are under: ${DATA_DIR}/${OUT_DIR}"

