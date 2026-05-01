#!/usr/bin/env python3
"""Tokenize an extra Hugging Face source into a standalone GPT-2 train.bin.

This script is intentionally simple:
- stream a single HF dataset
- optionally shuffle the stream with a buffer
- tokenize with GPT-2 tiktoken
- stop once roughly target_tokens are written

Outputs:
- train.bin
- metadata.json
"""

import argparse
import json
import os
import time
from typing import Optional

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_HF_HOME = os.path.join(PROJECT_ROOT, ".hf_cache")
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(DEFAULT_HF_HOME, "datasets"))

DEFAULT_TEXT_FIELDS = ["text", "content", "article", "body"]


PRESETS = {
    "openwebtext": {
        "dataset": "Skylion007/openwebtext",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "output_dir": "/data/fengfei/cse251b-nanogpt/build-nanogpt/tokenized_sources/openwebtext_2b",
    },
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "text_field": "text",
        "output_dir": "/data/fengfei/cse251b-nanogpt/build-nanogpt/tokenized_sources/wikipedia_2b",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one standalone tokenized source for v8.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    parser.add_argument("--dataset", type=str, default=None, help="Override HF dataset name.")
    parser.add_argument("--subset", type=str, default=None, help="Override HF subset/config name.")
    parser.add_argument("--split", type=str, default=None, help="Override HF split.")
    parser.add_argument("--text_field", type=str, default=None, help="Override preferred text field.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--target_tokens", type=int, default=2_000_000_000)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no_streaming", action="store_true")
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    parser.add_argument("--progress_every_docs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_text_field(row: dict, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in row and isinstance(row[preferred], str):
        return row[preferred]
    for k in DEFAULT_TEXT_FIELDS:
        if k in row and isinstance(row[k], str):
            return row[k]
    return None


def make_gpt2_tokenizer():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def main():
    args = parse_args()
    preset = dict(PRESETS[args.preset])
    dataset_name = args.dataset or preset["dataset"]
    subset = args.subset if args.subset is not None else preset["subset"]
    split = args.split or preset["split"]
    text_field = args.text_field or preset["text_field"]
    output_dir = args.output_dir or preset["output_dir"]
    streaming = not args.no_streaming

    if os.path.exists(output_dir) and os.listdir(output_dir) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty. Use --overwrite or choose a new output_dir.")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Preparing preset: {args.preset}")
    print(f"dataset={dataset_name} subset={subset} split={split}")
    print(f"output_dir={output_dir}")
    print(f"target_tokens={args.target_tokens:,}")
    print(f"streaming={streaming} shuffle_buffer_size={args.shuffle_buffer_size:,}")

    from datasets import load_dataset

    ds = load_dataset(dataset_name, name=subset, split=split, streaming=streaming)
    if args.shuffle_buffer_size > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    enc = make_gpt2_tokenizer()
    eot_id = 50256

    seen_docs = 0
    kept_docs = 0
    removed_empty = 0
    total_tokens = 0
    t0 = time.time()
    last_progress = t0

    train_bin = os.path.join(output_dir, "train.bin")
    with open(train_bin, "wb") as f:
        for row in ds:
            seen_docs += 1
            text = choose_text_field(row, text_field)
            if text is None:
                continue
            text = text.strip()
            if not text:
                removed_empty += 1
                continue

            ids = enc.encode_ordinary(text)
            arr = np.asarray([eot_id] + ids, dtype=np.uint16)
            arr.tofile(f)

            kept_docs += 1
            total_tokens += int(arr.size)

            if seen_docs % args.progress_every_docs == 0:
                now = time.time()
                dt = now - last_progress
                last_progress = now
                print(
                    f"[{args.preset}] seen={seen_docs:,} kept={kept_docs:,} "
                    f"tokens={total_tokens:,} removed_empty={removed_empty:,} dt={dt:.1f}s",
                    flush=True,
                )

            if total_tokens >= args.target_tokens:
                break

    elapsed = time.time() - t0
    meta = {
        "preset": args.preset,
        "dataset": dataset_name,
        "subset": subset,
        "split": split,
        "text_field": text_field,
        "target_tokens": args.target_tokens,
        "actual_tokens": total_tokens,
        "seen_docs": seen_docs,
        "kept_docs": kept_docs,
        "removed_empty": removed_empty,
        "tokenizer_backend": "gpt2_tiktoken",
        "tokenizer_vocab_size_actual": 50257,
        "eot_id": eot_id,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "seed": args.seed,
        "outputs": {
            "train_bin": train_bin,
        },
        "elapsed_seconds": elapsed,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"actual_tokens: {total_tokens:,}")
    print(f"kept_docs:     {kept_docs:,}")
    print(f"metadata:      {meta_path}")


if __name__ == "__main__":
    main()
