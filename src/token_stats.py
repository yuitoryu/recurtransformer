"""Summarize token counts in JSONL with one token ID array per line.

Usage: python -B -m src.token_stats data/tokenized/train/train.jsonl
Blank lines are ignored; empty arrays count as zero-token samples.
std is the population standard deviation (ddof=0).
"""

import argparse
import json
import math
from pathlib import Path


def token_stats(path):
    count = total = 0
    mean = m2 = 0.0
    minimum = maximum = None

    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            tokens = json.loads(line)
            if not isinstance(tokens, list):
                raise ValueError(f"{path}:{line_number}: expected a token ID array")
            length = len(tokens)
            count += 1
            total += length
            minimum = length if minimum is None else min(minimum, length)
            maximum = length if maximum is None else max(maximum, length)
            delta = length - mean
            mean += delta / count
            m2 += delta * (length - mean)

    if count == 0:
        raise ValueError(f"No samples found in {path}")

    return {
        "count": count,
        "total_tokens": total,
        "mean": mean,
        "min": minimum,
        "max": maximum,
        "std": math.sqrt(m2 / count),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL dataset path")
    args = parser.parse_args()
    print(json.dumps(token_stats(args.path), indent=2))


if __name__ == "__main__":
    main()
