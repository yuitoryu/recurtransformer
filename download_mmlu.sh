#!/usr/bin/env bash
# Run in the recur environment: bash download_mmlu.sh
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Share the cache used by src/evaluation.py; keep all files in the project.
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$PWD/.cache"
export TMPDIR="$PWD/.cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

python -B - <<'PY'
import os
from datasets import load_dataset

dataset = load_dataset("cais/mmlu", "all", cache_dir=os.environ["HF_DATASETS_CACHE"])
dataset.save_to_disk("data/mmlu")
for split, rows in dataset.items():
    print(f"{split}: {len(rows):,} examples")
print(f"MMLU saved to {os.path.abspath('data/mmlu')}")
PY
