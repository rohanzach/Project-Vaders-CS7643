#!/bin/bash
set -e
set -o pipefail

cd "$(dirname "$0")"

export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
PY=./venv312/bin/python
DATA_DIR=./data
LS_DIR="$DATA_DIR/LibriSpeech"

mkdir -p "$DATA_DIR"

echo "=== [1/4] Downloading LibriSpeech splits ==="
for split in dev-clean train-clean-100; do
  if [ -d "$LS_DIR/$split" ]; then
    echo "  $split already present"
  else
    echo "  downloading $split ..."
    curl -L -o "$DATA_DIR/$split.tar.gz" "https://www.openslr.org/resources/12/$split.tar.gz"
    echo "  extracting $split ..."
    tar -xzf "$DATA_DIR/$split.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/$split.tar.gz"
  fi
done

echo "=== [2/4] Training (extraction + 20 epochs) ==="
$PY train_lightweight.py

echo "=== [3/4] Evaluating via evaluate_cloning.py ==="
$PY evaluate_cloning.py

echo "=== [4/4] Done ==="
