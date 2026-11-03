"""Download GPT-2-counted tokens to data/train and data/valid, keeping whole documents.

Run from the project root: python -B src/download_data.py
Defaults: 2B training tokens and 10M validation tokens from separate source shards.
Validation is downloaded first; training excludes exact validation texts.
Parquet outputs and a small reuse manifest are written; HTTP buffering stays in memory.
"""

import argparse
import json
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


def file_stamp(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def download(tokens=2_000_000_000, split="train", exclude_sources=()):
    output = PROJECT_ROOT / "data" / split
    output.mkdir(parents=True, exist_ok=True)
    exclude_sources = set(exclude_sources)
    valid_texts = set()
    valid_files = []
    if split == "train":
        valid_files = sorted((PROJECT_ROOT / "data" / "valid").glob("*.parquet"))
        for path in valid_files:
            exclude_sources.add(SOURCE_PREFIX + path.name)
    manifest_path = output / "download_manifest.json"
    context = {"version": 1, "repo": REPO_ID, "prefix": SOURCE_PREFIX,
               "split": split, "exclude_sources": sorted(exclude_sources),
               "validation": {path.name: file_stamp(path) for path in valid_files}}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = manifest.get("shards", {}) if manifest.get("context") == context else {}

    def remember(source, destination, count, documents, remaining, finished):
        records[source] = {"tokens": count, "documents": documents,
                           "remaining": remaining, "finished": finished,
                           "file": file_stamp(destination) if destination.exists() else None}
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"context": context, "shards": records}), encoding="utf-8")
        temporary.replace(manifest_path)

    valid_loaded = False
    paths = sorted(
        path for path in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        if path.startswith(SOURCE_PREFIX) and path.endswith(".parquet")
        and path not in exclude_sources
    )
    # Validation starts at the other end of the sample.
    if split == "valid":
        paths.reverse()
    fs = HfFileSystem()
    used_sources = set()
    written_files = set()
    total_tokens = 0
    total_documents = 0
    next_report = 100_000_000
    for source in paths:
        used_sources.add(source)
        destination = output / Path(source).name
        remaining = tokens - total_tokens
        record = records.get(source)
        if (record is not None
                and record["file"] == (file_stamp(destination) if destination.exists() else None)
                and record["tokens"] <= remaining
                and (not record["finished"] or record["remaining"] == remaining)):
            total_tokens += record["tokens"]
            total_documents += record["documents"]
            if record["file"] is not None:
                written_files.add(destination)
            print(f"Reusing {destination.name}: {total_tokens:,}/{tokens:,} tokens", flush=True)
            if record["finished"] or total_tokens == tokens:
                break
            continue
        if not valid_loaded:
            for path in valid_files:
                for batch in pq.ParquetFile(path).iter_batches(columns=["text"]):
                    valid_texts.update(batch.column(0).to_pylist())
            valid_loaded = True
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
                    overlaps = valid_texts and any(
                        text in valid_texts
                        for batch in existing.iter_batches(columns=["text"])
                        for text in batch.column(0).to_pylist()
                    )
                    if total_tokens + count <= tokens and not overlaps:
                        remember(source, destination, count, existing.metadata.num_rows, remaining, False)
                        written_files.add(destination)
                        total_tokens += count
                        total_documents += existing.metadata.num_rows
                        print(f"Reusing {destination.name}: {total_tokens:,}/{tokens:,} tokens", flush=True)
                        if total_tokens == tokens:
                            break
                        continue

            print(f"Reading {source}; {tokens - total_tokens:,} tokens remaining", flush=True)
            writer = None
            finished = False
            start_tokens, start_documents = total_tokens, total_documents
            try:
                for batch in parquet.iter_batches(batch_size=1000, use_threads=False):
                    if valid_texts:
                        texts = batch.column(batch.schema.get_field_index("text")).to_pylist()
                        keep = [text not in valid_texts for text in texts]
                        if not all(keep):
                            batch = batch.filter(pa.array(keep))
                    if not batch.num_rows:
                        continue
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
                written_files.add(destination)
            elif destination.exists():
                destination.unlink()
            remember(source, destination, total_tokens - start_tokens,
                     total_documents - start_documents, remaining, finished)
            print(f"Saved {destination.name}: {total_tokens:,}/{tokens:,} tokens", flush=True)
            if finished:
                break
    else:
        raise RuntimeError("The sample was exhausted before reaching the token budget.")

    # Old shards (including a now fully filtered shard) must not enter the split.
    for path in output.glob("*.parquet"):
        if path not in written_files:
            path.unlink()

    print(
        f"Complete: {total_tokens:,} GPT-2 tokens, {total_documents:,} documents; "
        f"{tokens - total_tokens:,} tokens below budget. Output: {output}",
        flush=True,
    )
    return used_sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2_000_000_000)
    parser.add_argument("--valid-tokens", type=int, default=10_000_000,
                        help="validation token budget (default: 10000000; 0 skips validation download, "
                             "but training still excludes existing validation texts)")
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.valid_tokens < 0:
        parser.error("--valid-tokens must be non-negative")
    if args.valid_tokens:
        download(args.valid_tokens, split="valid")
    download(args.tokens)
