#!/usr/bin/env bash
# Run: bash eval.sh [extra evaluation arguments]
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
    for benchmark in mmlu piqa arc_easy hellaswag winogrande lambada; do
        shots=(0 5)
        if [[ "$benchmark" == lambada || "$benchmark" == mmlu ]]; then
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
done
exit "$status"
