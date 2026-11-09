#!/usr/bin/env bash
# Run in your Python environment: bash eval.sh [extra evaluation arguments]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
shopt -s nullglob

checkpoints=(runs/*/checkpoints/best.pt)
if ((${#checkpoints[@]} == 0)); then
    echo "No checkpoints/best.pt found under runs/" >&2
    exit 1
fi

status=0
for checkpoint in "${checkpoints[@]}"; do
    run_dir="${checkpoint%/checkpoints/best.pt}"
    echo "Evaluating $run_dir (MMLU 5-shot)"
    if ! python -B -m src.evaluation --run-dir "$run_dir" \
        --benchmark mmlu --num-fewshot 5 "$@"; then
        echo "Evaluation failed: $run_dir" >&2
        status=1
    fi
done
exit "$status"
