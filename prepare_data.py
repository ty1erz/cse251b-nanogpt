"""
Multi-domain data prep for the CSE 251B nanoGPT competition.

Tokenizes a HuggingFace dataset with the GPT-2 BPE and writes uint16 .npy
shards (same format as fineweb.py / DataLoaderLite expects).

Sources mirror the example eval blend (FineWeb-Edu 50%, Wikipedia 20%,
science 15%, books 15%). FineWeb-Edu is already prepared by fineweb.py;
this script handles the other three.

Usage:
    python prepare_data.py --source wikipedia
    python prepare_data.py --source science
    python prepare_data.py --source books

Output:
    ../data/<source>/<source>_<shard>.npy
"""

import argparse
import os
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Source registry. Each entry tells us which HF dataset/config/text-field to
# read and how many tokens to keep. Token caps are sized so that 5+ epochs of
# the target Phase-2 mix (~6B tokens, ratios 50/20/15/15) stay covered.

SOURCES = {
    "wikipedia": dict(
        repo="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        text_field="text",
        target_tokens=int(4.0e9),  # 4B tokens — supports 20B training at 20% mix
    ),
    "science": dict(
        # peS2o v2 / armanc/scientific_papers are script-based; HF datasets >= 2.15
        # disabled script loaders. open-web-math is parquet, ~14B tokens of
        # math + science web content, well-curated. Field: "text".
        repo="open-web-math/open-web-math",
        config=None,
        split="train",
        text_field="text",
        target_tokens=int(3.0e9),  # 3B tokens — supports 20B training at 15% mix
    ),
    "books": dict(
        repo="sedthh/gutenberg_english",
        config=None,
        split="train",
        text_field="TEXT",
        target_tokens=int(3.0e9),  # 3B tokens — supports 20B training at 15% mix
    ),
}

SHARD_SIZE = int(1e8)  # 100M tokens per shard

# Tokenizer — module level so multiprocessing workers can pickle the function.
_enc = tiktoken.get_encoding("gpt2")
_eot = _enc._special_tokens["<|endoftext|>"]


def tokenize(text: str) -> np.ndarray:
    """One document → uint16 token array, preceded by an EOT delimiter."""
    if not isinstance(text, str):
        # some Gutenberg rows have multi-paragraph lists
        text = "\n\n".join(text) if isinstance(text, list) else str(text)
    ids = [_eot]
    ids.extend(_enc.encode_ordinary(text))
    arr = np.array(ids, dtype=np.int64)
    assert (arr >= 0).all() and (arr < 2**16).all(), "token id out of uint16 range"
    return arr.astype(np.uint16)


def text_iter(ds, field):
    """Yield raw strings from a streaming HF dataset, robust to list-valued fields."""
    for ex in ds:
        if field not in ex:
            # try a few fallbacks
            for alt in ("text", "TEXT", "content", "raw_text"):
                if alt in ex:
                    field = alt
                    break
            else:
                raise KeyError(f"none of expected text fields found; keys={list(ex.keys())}")
        val = ex[field]
        if isinstance(val, list):
            val = "\n\n".join(str(v) for v in val)
        yield val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"),
                    help="output root; shards go to <out>/<source>/")
    ap.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    ap.add_argument("--target_tokens", type=int, default=None,
                    help="override default token cap for this source")
    ap.add_argument("--nprocs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    cfg = SOURCES[args.source]
    target = args.target_tokens or cfg["target_tokens"]
    out_dir = os.path.abspath(os.path.join(args.out, args.source))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{args.source}] streaming {cfg['repo']}"
          + (f" ({cfg['config']})" if cfg["config"] else ""))
    print(f"[{args.source}] target = {target:,} tokens, shards = {args.shard_size:,} tokens each")
    print(f"[{args.source}] output: {out_dir}")

    if cfg["config"]:
        ds = load_dataset(cfg["repo"], cfg["config"],
                          split=cfg["split"], streaming=True)
    else:
        ds = load_dataset(cfg["repo"], split=cfg["split"], streaming=True)

    shard_idx = 0
    buf = np.empty(args.shard_size, dtype=np.uint16)
    pos = 0
    total = 0

    pbar = tqdm(total=target, unit="tok", unit_scale=True, desc=args.source)

    with mp.Pool(args.nprocs) as pool:
        for toks in pool.imap(tokenize, text_iter(ds, cfg["text_field"]), chunksize=8):
            if total >= target:
                break

            i = 0
            while i < len(toks):
                room = args.shard_size - pos
                take = min(room, len(toks) - i)
                buf[pos:pos + take] = toks[i:i + take]
                pos += take
                i += take
                if pos >= args.shard_size:
                    fn = os.path.join(out_dir, f"{args.source}_{shard_idx:06d}.npy")
                    np.save(fn, buf)
                    shard_idx += 1
                    pos = 0

            total += len(toks)
            pbar.update(len(toks))

    if pos > 0:
        fn = os.path.join(out_dir, f"{args.source}_{shard_idx:06d}.npy")
        np.save(fn, buf[:pos])
        shard_idx += 1

    pbar.close()
    print(f"[{args.source}] done: {total:,} tokens across {shard_idx} shards")


if __name__ == "__main__":
    main()
