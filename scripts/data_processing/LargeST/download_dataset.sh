#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data/LargeST/raw"

if [ -d "$DATA_DIR" ] && [ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "LargeST data already exists. Skipping download."
    exit 0
fi

mkdir -p "$DATA_DIR"
pushd "$DATA_DIR" > /dev/null
echo "Downloading LargeST dataset..."
rm -f "$PROJECT_ROOT/largest.zip" || true
uv run --project "$PROJECT_ROOT" kaggle datasets download liuxu77/largest -p .
unzip largest.zip
rm largest.zip
popd > /dev/null # Return to project root
