#!/usr/bin/env bash
# Run in the recur environment:
#   bash eval_model.sh runs/vanilla-test [extra evaluation arguments]
# All benchmarks use 0/5-shot, except LAMBADA (0-shot only).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (($# == 0)); then
    echo "Usage: bash eval_model.sh RUN_DIR [extra evaluation arguments]" >&2
    exit 1
fi
run_dir="$1"
shift

if [[ ! -f "$run_dir/checkpoints/best.pt" ]]; then
    echo "Checkpoint not found: $run_dir/checkpoints/best.pt" >&2
    exit 1
fi

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$PWD/.cache"
export TMPDIR="$PWD/.cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

status=0
for benchmark in mmlu piqa arc_easy hellaswag winogrande lambada; do
    shots=(0 5)
    if [[ "$benchmark" == lambada ]]; then
        shots=(0)
    fi
    for shot in "${shots[@]}"; do
        echo "Evaluating $run_dir ($benchmark ${shot}-shot)"
        if ! python -B -m src.evaluation "$@" --run-dir "$run_dir" \
            --benchmark "$benchmark" --num-fewshot "$shot"; then
            echo "Evaluation failed: $run_dir ($benchmark ${shot}-shot)" >&2
            status=1
        fi
    done
done
exit "$status"
