#!/usr/bin/env bash
# Run in your Python environment: bash download_benchmarks.sh [benchmark ...]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$PWD/.cache"
export TMPDIR="$PWD/.cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

python -B - "$@" <<'PY'
import argparse
import os
from pathlib import Path

from datasets import load_dataset

sources = {
    # Parquet version also used by lm-evaluation-harness; no legacy loading script.
    "piqa": ("baber/piqa", None),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy"),
    "hellaswag": ("Rowan/hellaswag", None),
    "lambada": ("EleutherAI/lambada_openai", "default"),
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
}
parser = argparse.ArgumentParser(description="Download benchmarks into data/<benchmark>.")
parser.add_argument("benchmarks", nargs="*", choices=list(sources), default=list(sources))
args = parser.parse_args()

for name in args.benchmarks:
    repo, config = sources[name]
    destination = Path("data") / name
    print(f"Downloading {name}: {repo}", flush=True)
    dataset = load_dataset(repo, config, cache_dir=os.environ["HF_DATASETS_CACHE"])
    dataset.save_to_disk(str(destination))
    counts = ", ".join(f"{split}={len(rows):,}" for split, rows in dataset.items())
    print(f"Saved to {destination.resolve()} ({counts})", flush=True)
PY
