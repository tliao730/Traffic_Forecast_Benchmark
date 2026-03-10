#!/bin/bash

DATA_DIR="data/LargeST/raw"

if [ -d "$DATA_DIR" ]; then
    echo "LargeST data already exists. Skipping download."
    exit 0
fi

mkdir -p "$DATA_DIR"
pushd "$DATA_DIR"
echo "Downloading LargeST dataset..."
uv run kaggle datasets download liuxu77/largest
unzip largest.zip
rm largest.zip
popd # Return to project root
