"""
Train a custom BPE tokenizer (e.g., vocab_size=16000) for nanoGPT-style training.

Example:
python train_bpe_tokenizer.py --input_glob "data/text/*.txt" --vocab_size 16000 --output_dir tokenizer_16k
"""

import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for custom GPT vocab sizes.")
    parser.add_argument("--input_glob", type=str, required=True, help="Glob pattern for training text files.")
    parser.add_argument("--vocab_size", type=int, default=16000, help="Target tokenizer vocabulary size.")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum token frequency.")
    parser.add_argument("--output_dir", type=str, default="tokenizer_16k", help="Directory to save tokenizer files.")
    parser.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"], help="Special tokens to reserve.")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched input_glob: {args.input_glob}")

    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
    except ImportError as e:
        raise ImportError(
            "Please install `tokenizers` first: pip install tokenizers"
        ) from e

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_tokens + ["<unk>"],
        show_progress=True,
    )

    print(f"Training tokenizer on {len(files)} files...")
    tokenizer.train(files=files, trainer=trainer)

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to: {tokenizer_path}")
    print("Use the tokenizer externally to create token IDs, then write .bin/.npy shards for train_gpt2.py.")


if __name__ == "__main__":
    main()
