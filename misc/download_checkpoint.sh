#!/usr/bin/env bash
# Download a small NVFP4 checkpoint (ModelOpt export, ~1 GB) into data/.
# Usage: ./download_checkpoint.sh
set -euo pipefail

REPO="nvidia/Nemotron-3-Embed-1B-NVFP4"
FILES=("model.safetensors" "hf_quant_config.json" "config.json")
DEST="data"

mkdir -p "$DEST"

for file in "${FILES[@]}"; do
    if [[ -f "$DEST/$file" ]]; then
        echo "skip: $file already exists"
        continue
    fi
    echo "download: $file"
    curl --fail --location --progress-bar \
        --output "$DEST/$file.part" \
        "https://huggingface.co/$REPO/resolve/main/$file"
    mv "$DEST/$file.part" "$DEST/$file"
done

echo "done: files are in $DEST"
