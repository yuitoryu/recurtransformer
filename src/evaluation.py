"""Evaluate benchmarks using generated answers.

Run from the project root:
    python -B -m src.evaluation --run-dir runs/vanilla-test \
        --benchmark mmlu --num-fewshot 5 --batch-size 8

Print an accuracy table under runs/ (missing results are null):
    python -B -m src.evaluation --summary

Export the table to evaluation_summary.csv in the current directory:
    python -B -m src.evaluation --summary --csv
    python -B -m src.evaluation --summary --csv results/accuracy.csv

Add future benchmarks as evaluation functions and entries in BENCHMARKS.
MMLU defaults to 5-shot; other tasks default to 0-shot. LAMBADA is 0-shot only.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from .generate import CACHE_DIR, generate, generate_batch, load_model_and_tokenizer
from .common import PROJECT_ROOT, project_path, write_json

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


def _context_length(model, max_new_tokens, limit, max_seq_len):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    length = model.max_seq_len if max_seq_len is None else max_seq_len
    if not 1 <= length <= model.max_seq_len:
        raise ValueError(f"max_seq_len must be between 1 and {model.max_seq_len}")
    return length


def _evaluate_samples(model, tokenizer, samples, benchmark, num_fewshot,
                      max_new_tokens, limit, max_seq_len, batch_size=1):
    if not samples:
        raise ValueError("No evaluation questions selected")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    predictions = []
    for start in tqdm(range(0, len(samples), batch_size), desc=benchmark, unit="batches"):
        batch = samples[start:start + batch_size]
        if batch_size == 1:
            outputs = [generate(model, tokenizer, batch[0]["prompt"],
                                max_new_tokens=max_new_tokens, temperature=0,
                                max_seq_len=max_seq_len)]
        else:
            outputs = generate_batch(model, tokenizer, [sample["prompt"] for sample in batch],
                                     max_new_tokens=max_new_tokens, temperature=0,
                                     max_seq_len=max_seq_len)
        for sample, output in zip(batch, outputs):
            record = sample.copy()
            labels = record.pop("labels", None)
            prompt_tokens = len(tokenizer.encode(record["prompt"], add_special_tokens=False, verbose=False))
            if labels is None:  # LAMBADA: exact first-word match, including case/punctuation.
                words = output.split()
                prediction = words[0] if words else None
            else:
                match = re.match(r"^\s*([A-Z])\b", output, flags=re.IGNORECASE)
                prediction = match.group(1).upper() if match else None
                if prediction is not None and prediction not in labels:
                    prediction = None
            record.update(prompt_tokens=prompt_tokens, raw_output=output, prediction=prediction,
                          correct=prediction == record["target"], truncated=prompt_tokens > max_seq_len)
            predictions.append(record)
    summary = {
        "benchmark": benchmark, "evaluation": "generation",
        "num_fewshot": num_fewshot, "max_new_tokens": max_new_tokens,
        "temperature": 0, "max_seq_len": max_seq_len, "batch_size": batch_size,
        "model_max_seq_len": model.max_seq_len, "limit": limit,
        **_metrics(predictions),
    }
    return {"summary": summary, "predictions": predictions}


def evaluate_mmlu(model, tokenizer, num_fewshot=5, max_new_tokens=4,
                  subjects=None, limit=None, max_seq_len=None, batch_size=1):
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
    max_seq_len = _context_length(model, max_new_tokens, limit, max_seq_len)

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

    samples = [{"index": index, "subject": row["subject"], "labels": "ABCD",
                "prompt": prefixes[row["subject"]] + _format_mmlu_example(row),
                "target": "ABCD"[row["answer"]]} for index, row in samples]
    result = _evaluate_samples(model, tokenizer, samples, "mmlu", num_fewshot,
                               max_new_tokens, limit, max_seq_len, batch_size)
    by_subject = defaultdict(list)
    for record in result["predictions"]:
        by_subject[record["subject"]].append(record)
    result["summary"].update(
        dataset="cais/mmlu", split="test", fewshot_split="dev" if num_fewshot else None,
        subjects=sorted(selected_subjects) if subjects is not None else None,
        per_subject={subject: _metrics(rows) for subject, rows in sorted(by_subject.items())},
    )
    return result


CHOICE_DATASETS = {
    "piqa": ("baber/piqa", "validation"),
    "arc_easy": ("allenai/ai2_arc", "test"),
    "hellaswag": ("Rowan/hellaswag", "validation"),
    "winogrande": ("allenai/winogrande", "validation"),
}


def _load_local(benchmark):
    directory = PROJECT_ROOT / "data" / benchmark
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {directory}; run: bash download_benchmarks.sh {benchmark}")
    return load_from_disk(str(directory))


def _choice_example(benchmark, row):
    if benchmark == "piqa":
        question, choices, answer = row["goal"], [row["sol1"], row["sol2"]], row["label"]
    elif benchmark == "arc_easy":
        question, choices = row["question"], row["choices"]["text"]
        answer = row["choices"]["label"].index(row["answerKey"])
    elif benchmark == "hellaswag":
        question = "Choose the most plausible continuation:\n" + row["ctx"]
        choices, answer = row["endings"], int(row["label"])
    else:  # winogrande
        question = "Fill in the blank:\n" + row["sentence"]
        choices, answer = [row["option1"], row["option2"]], int(row["answer"]) - 1
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, choices))
    return f"{question.strip()}\n{options}\nAnswer:", labels[answer], labels


def _evaluate_choices(benchmark, model, tokenizer, num_fewshot, max_new_tokens, limit, max_seq_len, batch_size=1):
    if num_fewshot < 0:
        raise ValueError("num_fewshot must be non-negative")
    max_seq_len = _context_length(model, max_new_tokens, limit, max_seq_len)
    dataset = _load_local(benchmark)
    prefix = "Answer each multiple choice question with the option letter.\n\n"
    if num_fewshot:
        if num_fewshot > len(dataset["train"]):
            raise ValueError("num_fewshot exceeds the number of training examples")
        for index in range(num_fewshot):
            prompt, target, _ = _choice_example(benchmark, dataset["train"][index])
            prefix += f"{prompt} {target}\n\n"
    repo, split = CHOICE_DATASETS[benchmark]
    rows = dataset[split]
    count = len(rows) if limit is None else min(limit, len(rows))
    samples = []
    for index in range(count):
        prompt, target, labels = _choice_example(benchmark, rows[index])
        samples.append({"index": index, "prompt": prefix + prompt, "target": target, "labels": labels})
    result = _evaluate_samples(model, tokenizer, samples, benchmark, num_fewshot,
                               max_new_tokens, limit, max_seq_len, batch_size)
    result["summary"].update(dataset=repo, split=split,
                             fewshot_split="train" if num_fewshot else None)
    return result


def evaluate_piqa(model, tokenizer, num_fewshot=0, max_new_tokens=4, limit=None, max_seq_len=None, batch_size=1):
    return _evaluate_choices("piqa", model, tokenizer, num_fewshot, max_new_tokens, limit, max_seq_len, batch_size)


def evaluate_arc_easy(model, tokenizer, num_fewshot=0, max_new_tokens=4, limit=None, max_seq_len=None, batch_size=1):
    return _evaluate_choices("arc_easy", model, tokenizer, num_fewshot, max_new_tokens, limit, max_seq_len, batch_size)


def evaluate_hellaswag(model, tokenizer, num_fewshot=0, max_new_tokens=4, limit=None, max_seq_len=None, batch_size=1):
    return _evaluate_choices("hellaswag", model, tokenizer, num_fewshot, max_new_tokens, limit, max_seq_len, batch_size)


def evaluate_winogrande(model, tokenizer, num_fewshot=0, max_new_tokens=4, limit=None, max_seq_len=None, batch_size=1):
    return _evaluate_choices("winogrande", model, tokenizer, num_fewshot, max_new_tokens, limit, max_seq_len, batch_size)


def evaluate_lambada(model, tokenizer, num_fewshot=0, max_new_tokens=16, limit=None, max_seq_len=None, batch_size=1):
    """Generate the last whitespace-delimited word; preserve target case/punctuation."""
    if num_fewshot != 0:
        raise ValueError("LAMBADA supports only num_fewshot=0; its local dataset has no training split")
    max_seq_len = _context_length(model, max_new_tokens, limit, max_seq_len)
    rows = _load_local("lambada")["test"]
    count = len(rows) if limit is None else min(limit, len(rows))
    samples = []
    for index in range(count):
        prompt, target = rows[index]["text"].rstrip().rsplit(None, 1)
        samples.append({"index": index, "prompt": prompt, "target": target})
    result = _evaluate_samples(model, tokenizer, samples, "lambada", 0,
                               max_new_tokens, limit, max_seq_len, batch_size)
    result["summary"].update(dataset="EleutherAI/lambada_openai", split="test", fewshot_split=None)
    return result


BENCHMARKS = {
    "mmlu": evaluate_mmlu, "piqa": evaluate_piqa, "arc_easy": evaluate_arc_easy,
    "hellaswag": evaluate_hellaswag, "winogrande": evaluate_winogrande, "lambada": evaluate_lambada,
}


def collect_results():
    """Collect accuracies per run, including missing benchmark/shot pairs."""
    evaluation_names = {
        f"{benchmark}_{shot}_shot"
        for benchmark in BENCHMARKS
        for shot in ((0,) if benchmark == "lambada" else (0, 5))
    }
    results = {}
    for run_dir in sorted((PROJECT_ROOT / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        summaries = {}
        for path in sorted((run_dir / "evaluation").glob("*.json")):
            summaries[path.stem] = json.loads(path.read_text(encoding="utf-8"))["accuracy"]
        results[run_dir.name] = summaries
        evaluation_names.update(summaries)
    return {
        run: {name: summaries.get(name) for name in sorted(evaluation_names)}
        for run, summaries in results.items()
    }


def _accuracy_table(results):
    """Build shared table cells for terminal and CSV output."""
    names = list(next(iter(results.values()), {}))
    headers = ["Model"] + [re.sub(r"_(\d+)_shot$", r" \1-shot", name) for name in names]
    rows = [
        [run] + ["null" if accuracies[name] is None else f"{accuracies[name]:.2%}" for name in names]
        for run, accuracies in results.items()
    ]
    return headers, rows


def print_accuracy_table(results):
    """Print one model per row and one benchmark/shot pair per column."""
    if not results:
        print("No runs found under runs/")
        return
    headers, rows = _accuracy_table(results)
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows)]

    def format_row(row):
        return " | ".join(cell.ljust(width) if index == 0 else cell.rjust(width)
                          for index, (cell, width) in enumerate(zip(row, widths)))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--summary", action="store_true",
                      help="print an accuracy table (percentages) under runs/; missing results are null")
    parser.add_argument("--csv", nargs="?", type=Path, const=Path("evaluation_summary.csv"),
                        metavar="PATH", help="with --summary, export CSV instead of printing the table; "
                        "default: ./evaluation_summary.csv (relative to the current directory)")
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="mmlu")
    mode.add_argument("--run-dir", type=Path,
                      help="experiment directory containing checkpoints/best.pt")
    parser.add_argument("--device", help="default: CUDA if available, otherwise CPU")
    parser.add_argument("--num-fewshot", type=int, help="default: 5 for MMLU, 0 otherwise")
    parser.add_argument("--max-new-tokens", type=int, help="default: 16 for LAMBADA, 4 otherwise")
    parser.add_argument("--max-seq-len", type=int,
                        help="context length; default and upper limit: checkpoint's max_seq_len")
    parser.add_argument("--subjects", nargs="+", help="MMLU only; default: all subjects")
    parser.add_argument("--limit", type=int, help="first N questions after subject filtering")
    parser.add_argument("--batch-size", "--batchnumber", type=int, default=1,
                        help="questions generated per batch (default: 1)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.csv is not None and not args.summary:
        parser.error("--csv requires --summary")
    if args.summary:
        if args.csv is not None:
            output_path = args.csv.resolve()
            if not output_path.is_relative_to(PROJECT_ROOT):
                parser.error("The CSV output path must be inside the project")
            headers, rows = _accuracy_table(collect_results())
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(headers)
                writer.writerows(rows)
            print(f"CSV: {output_path}")
        else:
            print_accuracy_table(collect_results())
        return 0
    if args.subjects is not None and args.benchmark != "mmlu":
        parser.error("--subjects is only supported for MMLU")
    num_fewshot = args.num_fewshot if args.num_fewshot is not None else (5 if args.benchmark == "mmlu" else 0)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (16 if args.benchmark == "lambada" else 4)
    if args.benchmark == "lambada" and num_fewshot != 0:
        parser.error("LAMBADA supports only --num-fewshot 0")

    run_dir = project_path(args.run_dir)
    checkpoint = run_dir / "checkpoints" / "best.pt"
    output_dir = (run_dir / "evaluation").resolve()
    output_name = f"{args.benchmark}_{num_fewshot}_shot"
    if not output_dir.is_relative_to(PROJECT_ROOT):
        parser.error("The experiment evaluation directory must be inside the project")

    model, tokenizer = load_model_and_tokenizer(checkpoint, device=args.device)
    options = dict(num_fewshot=num_fewshot, max_new_tokens=max_new_tokens,
                   limit=args.limit, max_seq_len=args.max_seq_len, batch_size=args.batch_size)
    if args.benchmark == "mmlu":
        options["subjects"] = args.subjects
    result = BENCHMARKS[args.benchmark](model, tokenizer, **options)
    summary = result["summary"]
    summary.update(run_dir=str(run_dir), checkpoint=str(checkpoint), model=type(model).__name__,
                   device=str(next(model.parameters()).device))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{output_name}_predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in result["predictions"]:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / f"{output_name}.json", summary)
    print(f"{args.benchmark} generation: accuracy={summary['accuracy']:.4f}, "
          f"total={summary['total']}, invalid={summary['invalid']}, "
          f"truncated={summary['truncated']}")
    print(f"Results: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
