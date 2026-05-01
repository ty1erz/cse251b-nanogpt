"""Prepare training data by concatenating text sources and tokenizing them."""

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np

# Default Hugging Face caches to the project directory so dataset downloads
# stay under /data/fengfei/cse251b-nanogpt unless explicitly overridden.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_HF_HOME = os.path.join(PROJECT_ROOT, ".hf_cache")
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(DEFAULT_HF_HOME, "datasets"))

DEFAULT_TEXT_FIELDS = ["text", "content", "article", "body"]


@dataclass
class DatasetStats:
    seen_docs: int = 0
    kept_docs: int = 0
    removed_empty: int = 0
    est_tokens_kept: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tokenized training data from one or more datasets.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke: tiny default run; full: larger template config.",
    )
    parser.add_argument(
        "--dataset_mix_config",
        type=str,
        default=None,
        help="Path to JSON mixture config. If omitted, built-in smoke/full defaults are used.",
    )
    parser.add_argument(
        "--max_documents_per_dataset",
        type=int,
        default=5000,
        help="Global cap per dataset entry (can be overridden by config entry). Use 0 for no cap.",
    )
    parser.add_argument(
        "--target_num_tokens",
        type=int,
        default=500_000,
        help="Approximate target total tokens. Use 0 to collect all available documents.",
    )
    parser.add_argument("--output_dir", type=str, default="prepared_data_smoke")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chars_per_token_est", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def built_in_mix(mode: str) -> Dict:
    if mode == "full":
        return {
            "description": "Full two-dataset mixture.",
            "datasets": [
                {
                    "name": "fineweb_edu_main",
                    "source": "hf",
                    "dataset": "HuggingFaceFW/fineweb-edu",
                    "subset": "sample-10BT",
                    "split": "train",
                    "text_field": "text",
                    "max_docs": 0,
                    "streaming": True,
                },
                {
                    "name": "cosmopedia_textbook",
                    "source": "hf",
                    "dataset": "HuggingFaceTB/cosmopedia",
                    "subset": "web_samples_v2",
                    "split": "train",
                    "text_field": "text",
                    "max_docs": 0,
                    "streaming": True,
                },
            ],
        }
    return {
        "description": "Small smoke-test run.",
        "datasets": [
            {
                "name": "fineweb_edu_main",
                "source": "hf",
                "dataset": "HuggingFaceFW/fineweb-edu",
                "subset": "sample-10BT",
                "split": "train",
                "text_field": "text",
                "max_docs": 1500,
                "streaming": True,
            },
            {
                "name": "cosmopedia_textbook",
                "source": "hf",
                "dataset": "HuggingFaceTB/cosmopedia",
                "subset": "web_samples_v2",
                "split": "train",
                "text_field": "text",
                "max_docs": 1000,
                "streaming": True,
            },
        ],
    }


def load_mix_config(path: Optional[str], mode: str) -> Dict:
    if path is None:
        return built_in_mix(mode)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_text_field(row: Dict, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in row and isinstance(row[preferred], str):
        return row[preferred]
    for k in DEFAULT_TEXT_FIELDS:
        if k in row and isinstance(row[k], str):
            return row[k]
    return None


def iter_local_text_dir(path: str) -> Iterator[Dict]:
    files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                yield {"text": f.read()}
        except UnicodeDecodeError:
            with open(fp, "r", encoding="latin-1") as f:
                yield {"text": f.read()}


def iter_local_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_entry_rows(entry: Dict) -> Iterable[Dict]:
    source = entry.get("source", "hf")
    if source == "hf":
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("Please install datasets: pip install datasets") from e
        ds_name = entry["dataset"]
        subset = entry.get("subset")
        split = entry.get("split", "train")
        streaming = bool(entry.get("streaming", True))
        ds = load_dataset(ds_name, name=subset, split=split, streaming=streaming)
        return ds
    if source == "local_text_dir":
        return iter_local_text_dir(entry["path"])
    if source == "local_jsonl":
        return iter_local_jsonl(entry["path"])
    raise ValueError(f"Unknown source type: {source}")


def iter_texts_for_entry(
    entry: Dict,
    global_max_docs: int,
    target_tokens_for_entry: int,
    chars_per_token_est: float,
) -> Iterator[Tuple[str, DatasetStats]]:
    stats = DatasetStats()
    entry_max_docs = int(entry.get("max_docs", global_max_docs))
    if global_max_docs <= 0 and entry_max_docs <= 0:
        max_docs = None
    elif global_max_docs <= 0:
        max_docs = entry_max_docs
    elif entry_max_docs <= 0:
        max_docs = global_max_docs
    else:
        max_docs = min(entry_max_docs, global_max_docs)
    text_field = entry.get("text_field")
    progress_every = 10000
    last_progress_time = time.time()
    progress_name = entry.get("name", entry.get("dataset", "dataset"))

    est_token_budget = max(0, int(target_tokens_for_entry))
    est_tokens = 0

    for row in iter_entry_rows(entry):
        stats.seen_docs += 1
        text = choose_text_field(row, text_field)
        if text is None:
            continue
        text = text.strip()
        if not text:
            stats.removed_empty += 1
            continue
        stats.kept_docs += 1
        est = max(1, int(round(len(text) / chars_per_token_est)))
        est_tokens += est
        stats.est_tokens_kept = est_tokens
        yield text, stats

        if stats.seen_docs % progress_every == 0:
            now = time.time()
            elapsed = now - last_progress_time
            last_progress_time = now
            print(
                f"[{progress_name}] seen={stats.seen_docs:,} kept={stats.kept_docs:,} "
                f"est_tokens={stats.est_tokens_kept:,} "
                f"removed_empty={stats.removed_empty:,} "
                f"dt={elapsed:.1f}s",
                flush=True,
            )

        if max_docs is not None and stats.kept_docs >= max_docs:
            break
        if est_token_budget > 0 and est_tokens >= est_token_budget:
            break


def make_gpt2_tokenizer():
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError("Please install tiktoken: pip install tiktoken") from e
    return tiktoken.get_encoding("gpt2")


def tokenize_text(enc, text: str, eot_id: int) -> np.ndarray:
    ids = enc.encode_ordinary(text)
    return np.asarray([eot_id] + ids, dtype=np.uint16)


def main():
    args = parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty. Use --overwrite or a new output_dir.")
    os.makedirs(args.output_dir, exist_ok=True)

    mix = load_mix_config(args.dataset_mix_config, args.mode)
    entries = mix.get("datasets", [])
    if len(entries) == 0:
        raise ValueError("dataset mix config contains no datasets")

    print("Mixture entries:")
    for e in entries:
        print(f"- {e.get('name', e.get('dataset', 'dataset'))}")

    per_dataset_meta = []
    total_docs = 0
    total_tokens = 0

    train_bin = os.path.join(args.output_dir, "train.bin")
    tokenizer_json = None
    tokenizer_dir = None
    eot_id = 50256
    vocab_size = 50257
    enc = make_gpt2_tokenizer()
    print("Using GPT-2 tiktoken (vocab=50257).")

    with open(train_bin, "wb") as train_f:
        for entry in entries:
            name = entry.get("name", entry.get("dataset", "dataset"))
            tok_budget = int(args.target_num_tokens / len(entries)) if args.target_num_tokens > 0 else 0
            print(f"\nCollecting: {name} (est_token_budget={tok_budget})")
            stats = DatasetStats()
            entry_tokens = 0
            for text, stats in iter_texts_for_entry(
                entry=entry,
                global_max_docs=args.max_documents_per_dataset,
                target_tokens_for_entry=tok_budget,
                chars_per_token_est=args.chars_per_token_est,
            ):
                tokens = tokenize_text(enc, text, eot_id=eot_id)
                tokens.tofile(train_f)
                entry_tokens += int(tokens.size)
                total_tokens += int(tokens.size)
            total_docs += stats.kept_docs
            per_dataset_meta.append(
                {
                    "name": name,
                    "source": entry.get("source", "hf"),
                    "dataset": entry.get("dataset"),
                    "subset": entry.get("subset"),
                    "split": entry.get("split", "train"),
                    "seen_docs": stats.seen_docs,
                    "kept_docs": stats.kept_docs,
                    "est_tokens_kept": stats.est_tokens_kept,
                    "train_tokens": entry_tokens,
                    "removed_empty": stats.removed_empty,
                }
            )
            print(
                f"kept={stats.kept_docs} seen={stats.seen_docs} "
                f"est_tokens={stats.est_tokens_kept} train_tokens={entry_tokens} "
                f"removed_empty={stats.removed_empty}"
            )
    if total_docs == 0:
        raise RuntimeError("No documents left after filtering.")

    metadata = {
        "description": mix.get("description", ""),
        "mode": args.mode,
        "seed": args.seed,
        "dataset_mix_config": args.dataset_mix_config,
        "target_num_tokens_estimate": args.target_num_tokens,
        "tokenizer_vocab_size_actual": vocab_size,
        "tokenizer_backend": "gpt2_tiktoken",
        "tokenizer_json": tokenizer_json,
        "normalization": "strip only",
        "datasets": per_dataset_meta,
        "train_docs": total_docs,
        "train_tokens": total_tokens,
        "outputs": {
            "train_bin": train_bin,
            "tokenizer_dir": tokenizer_dir,
        },
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"train_docs:        {total_docs:,}")
    print(f"train.bin tokens: {total_tokens:,}")
    print(f"metadata:         {meta_path}")


if __name__ == "__main__":
    main()
