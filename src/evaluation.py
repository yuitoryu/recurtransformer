"""Evaluate benchmarks using generated answers.

Run from the project root:
    python -B -m src.evaluation --run-dir runs/vanilla-test \
        --benchmark mmlu --num-fewshot 5

Add future benchmarks as evaluation functions and entries in BENCHMARKS.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from .generate import CACHE_DIR, generate, load_model_and_tokenizer
from .train import PROJECT_ROOT, project_path, write_json

from datasets import load_dataset, load_from_disk
from tqdm import tqdm


def _load_mmlu(split):
    data_dir = PROJECT_ROOT / "data" / "mmlu"
    if data_dir.is_dir():
        return load_from_disk(str(data_dir))[split]
    return load_dataset("cais/mmlu", "all", split=split,
                        cache_dir=str(CACHE_DIR / "datasets"))


def _format_mmlu_example(example, include_answer=False):
    options = "\n".join(
        f"{letter}. {text}" for letter, text in zip("ABCD", example["choices"])
    )
    prompt = f"{example['question'].strip()}\n{options}\nAnswer:"
    if include_answer:
        prompt += f" {'ABCD'[example['answer']]}"
    return prompt


def _metrics(predictions):
    total = len(predictions)
    correct = sum(row["correct"] for row in predictions)
    invalid = sum(row["prediction"] is None for row in predictions)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "invalid": invalid,
        "invalid_rate": invalid / total,
        "truncated": sum(row["truncated"] for row in predictions),
    }


def evaluate_mmlu(model, tokenizer, num_fewshot=5, max_new_tokens=4,
                  subjects=None, limit=None, max_seq_len=None):
    """Return {'summary': ..., 'predictions': ...} for generation-based MMLU.

    subjects is a sequence of subject names, or None for all subjects. limit
    selects the first N test rows after filtering, retaining their original
    indices and order. Few-shot examples come only from the same subject's dev
    split. Overlong prompts use generate()'s usual left truncation; 'truncated'
    counts initial prompt overflow, not later sliding during generation.
    max_seq_len optionally limits context to at most the model's window.
    """
    if not 0 <= num_fewshot <= 5:
        raise ValueError("num_fewshot must be between 0 and 5")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    max_seq_len = model.max_seq_len if max_seq_len is None else max_seq_len
    if not 1 <= max_seq_len <= model.max_seq_len:
        raise ValueError(f"max_seq_len must be between 1 and {model.max_seq_len}")

    test = _load_mmlu("test")
    selected_subjects = set(subjects) if subjects is not None else set(test["subject"])
    unknown = selected_subjects - set(test["subject"])
    if unknown:
        raise ValueError(f"Unknown MMLU subjects: {', '.join(sorted(unknown))}")
    samples = [(index, row) for index, row in enumerate(test)
               if row["subject"] in selected_subjects]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No MMLU test questions selected")

    dev_by_subject = defaultdict(list)
    if num_fewshot:
        dev = _load_mmlu("dev")
        for row in dev:
            dev_by_subject[row["subject"]].append(row)

    prefixes = {}
    for subject in {row["subject"] for _, row in samples}:
        examples = dev_by_subject[subject][:num_fewshot]
        if len(examples) != num_fewshot:
            raise ValueError(f"Not enough dev examples for {subject}: need {num_fewshot}")
        header = ("The following are multiple choice questions (with answers) "
                  f"about {subject.replace('_', ' ')}.\n\n")
        prefixes[subject] = header + "".join(
            _format_mmlu_example(row, include_answer=True) + "\n\n" for row in examples
        )

    predictions = []
    by_subject = defaultdict(list)
    for index, row in tqdm(samples, desc="MMLU", unit="questions"):
        subject = row["subject"]
        prompt = prefixes[subject] + _format_mmlu_example(row)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        output = generate(model, tokenizer, prompt,
                          max_new_tokens=max_new_tokens, temperature=0,
                          max_seq_len=max_seq_len)
        match = re.match(r"^\s*([ABCD])\b", output, flags=re.IGNORECASE)
        prediction = match.group(1).upper() if match else None
        target = "ABCD"[row["answer"]]
        record = {
            "index": index,
            "subject": subject,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "raw_output": output,
            "prediction": prediction,
            "target": target,
            "correct": prediction == target,
            "truncated": prompt_tokens > max_seq_len,
        }
        predictions.append(record)
        by_subject[subject].append(record)

    summary = {
        "benchmark": "mmlu",
        "evaluation": "generation",
        "dataset": "cais/mmlu",
        "split": "test",
        "num_fewshot": num_fewshot,
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
        "max_seq_len": max_seq_len,
        "model_max_seq_len": model.max_seq_len,
        "subjects": sorted(selected_subjects) if subjects is not None else None,
        "limit": limit,
        **_metrics(predictions),
        "per_subject": {subject: _metrics(rows) for subject, rows in sorted(by_subject.items())},
    }
    return {"summary": summary, "predictions": predictions}


BENCHMARKS = {"mmlu": evaluate_mmlu}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="mmlu")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="experiment directory containing checkpoints/best.pt")
    parser.add_argument("--device", help="default: CUDA if available, otherwise CPU")
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int,
                        help="context length; default and upper limit: checkpoint's max_seq_len")
    parser.add_argument("--subjects", nargs="+", help="default: all subjects")
    parser.add_argument("--limit", type=int, help="first N questions after subject filtering")
    args = parser.parse_args(argv)

    run_dir = project_path(args.run_dir)
    checkpoint = run_dir / "checkpoints" / "best.pt"
    output_dir = (run_dir / "evaluation" / args.benchmark).resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT):
        parser.error("The experiment evaluation directory must be inside the project")

    model, tokenizer = load_model_and_tokenizer(checkpoint, device=args.device)
    result = BENCHMARKS[args.benchmark](
        model, tokenizer, num_fewshot=args.num_fewshot,
        max_new_tokens=args.max_new_tokens, subjects=args.subjects, limit=args.limit,
        max_seq_len=args.max_seq_len,
    )
    summary = result["summary"]
    summary.update(run_dir=str(run_dir), checkpoint=str(checkpoint), model=type(model).__name__,
                   device=str(next(model.parameters()).device))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in result["predictions"]:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / "summary.json", summary)
    print(f"{args.benchmark} generation: accuracy={summary['accuracy']:.4f}, "
          f"total={summary['total']}, invalid={summary['invalid']}, "
          f"truncated={summary['truncated']}")
    print(f"Results: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
