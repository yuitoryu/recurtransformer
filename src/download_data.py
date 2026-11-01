"""Download ~2B official GPT-2 tokens to data/train, keeping whole documents.

Run from the project root: python -B src/download_data.py
Only the output Parquet files are written; HTTP buffering stays in memory.
"""

import argparse
import os
from pathlib import Path

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "HuggingFaceFW/fineweb-edu"
SOURCE_PREFIX = "sample/10BT/"


def prefix_rows(counts, budget):
    cumulative = np.cumsum(counts, dtype=np.int64)
    rows = int(np.searchsorted(cumulative, budget, side="right"))
    return rows, int(cumulative[rows - 1]) if rows else 0


def download(tokens=2_000_000_000):
    output = PROJECT_ROOT / "data" / "train"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path for path in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        if path.startswith(SOURCE_PREFIX) and path.endswith(".parquet")
    )
    fs = HfFileSystem()
    total_tokens = 0
    total_documents = 0
    next_report = 100_000_000
    for source in paths:
        destination = output / Path(source).name
        # fsspec's readahead cache stores bytes in RAM, with no disk cache.
        with fs.open(
            f"datasets/{REPO_ID}/{source}", "rb",
            block_size=16 * 1024 * 1024, cache_type="readahead",
        ) as remote:
            parquet = pq.ParquetFile(remote)
            if destination.exists():
                try:
                    existing = pq.ParquetFile(destination)
                except pa.ArrowInvalid:
                    existing = None  # An interrupted direct write has no footer.
                if existing is not None and existing.metadata.num_rows == parquet.metadata.num_rows:
                    count = int(pq.read_table(destination, columns=["token_count"])["token_count"].to_numpy().sum())
                    if total_tokens + count <= tokens:
                        total_tokens += count
                        total_documents += existing.metadata.num_rows
                        print(f"Reusing {destination.name}: {total_tokens:,}/{tokens:,} tokens", flush=True)
                        if total_tokens == tokens:
                            break
                        continue

            print(f"Reading {source}; {tokens - total_tokens:,} tokens remaining", flush=True)
            writer = None
            finished = False
            try:
                for batch in parquet.iter_batches(batch_size=1000, use_threads=False):
                    counts = batch.column(batch.schema.get_field_index("token_count")).to_numpy()
                    rows, count = prefix_rows(counts, tokens - total_tokens)
                    if rows:
                        if writer is None:
                            writer = pq.ParquetWriter(destination, parquet.schema_arrow, compression="zstd")
                        writer.write_batch(batch.slice(0, rows))
                        total_tokens += count
                        total_documents += rows
                    if total_tokens >= next_report:
                        print(f"Progress: {total_tokens:,}/{tokens:,} tokens", flush=True)
                        next_report = (total_tokens // 100_000_000 + 1) * 100_000_000
                    if rows < batch.num_rows or total_tokens == tokens:
                        finished = True
                        break
            except BaseException:
                if writer is not None:
                    writer.close()
                    destination.unlink()
                raise
            if writer is not None:
                writer.close()
            print(f"Saved {destination.name}: {total_tokens:,}/{tokens:,} tokens", flush=True)
            if finished:
                break
    else:
        raise RuntimeError("The sample was exhausted before reaching the token budget.")

    print(
        f"Complete: {total_tokens:,} GPT-2 tokens, {total_documents:,} documents; "
        f"{tokens - total_tokens:,} tokens below budget. Output: {output}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2_000_000_000)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    download(args.tokens)
