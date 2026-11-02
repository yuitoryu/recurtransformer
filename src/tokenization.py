"""Tokenize one dataset section into a single JSONL file.

Run from the project root: python -B -m src.tokenization --section train
Default output: data/tokenized/<section>/<section>.jsonl
Each document gets an EOS token before being split into unpadded chunks.
"""

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(CACHE_DIR)

import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import GPT2TokenizerFast


def tokenize(section, data_dir, save_dir, seq_len=2048):
    paths = sorted(data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")

    tokenizer = GPT2TokenizerFast.from_pretrained(
        "openai-community/gpt2", cache_dir=str(CACHE_DIR)
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    destination = save_dir / f"{section}.jsonl"
    total_documents = 0
    total_sequences = 0
    total_tokens = 0

    with destination.open("w", encoding="utf-8") as output:
        for source in paths:
            with pq.ParquetFile(source) as parquet:
                with tqdm(
                    total=parquet.metadata.num_rows, desc=source.name, unit="docs"
                ) as progress:
                    for batch in parquet.iter_batches(
                        batch_size=256, columns=["text"]
                    ):
                        texts = batch.column(0).to_pylist()
                        encoded = tokenizer(
                            texts,
                            add_special_tokens=False,
                            truncation=False,
                            padding=False,
                            return_attention_mask=False,
                            return_token_type_ids=False,
                            verbose=False,
                        )["input_ids"]
                        for token_ids in encoded:
                            token_ids.append(tokenizer.eos_token_id)
                            total_tokens += len(token_ids)
                            for start in range(0, len(token_ids), seq_len):
                                chunk = token_ids[start : start + seq_len]
                                output.write(json.dumps(chunk) + "\n")
                                total_sequences += 1
                        total_documents += batch.num_rows
                        progress.update(batch.num_rows)

    print(
        f"Complete: {total_documents:,} documents, "
        f"{total_sequences:,} sequences, {total_tokens:,} tokens "
        f"(including EOS). Output: {destination}",
        flush=True,
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", required=True, choices=("train", "valid"))
    parser.add_argument("--data-dir", type=Path, help="default: data/<section>")
    parser.add_argument(
        "--save-dir", type=Path, help="default: data/tokenized/<section>"
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()
    if args.seq_len <= 0:
        parser.error("--seq-len must be positive")

    data_dir = args.data_dir if args.data_dir is not None else (
        PROJECT_ROOT / "data" / args.section
    )
    save_dir = args.save_dir if args.save_dir is not None else (
        PROJECT_ROOT / "data" / "tokenized" / args.section
    )
    tokenize(args.section, data_dir, save_dir, args.seq_len)


if __name__ == "__main__":
    main()
