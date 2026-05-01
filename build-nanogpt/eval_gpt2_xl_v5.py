#!/usr/bin/env python3
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    data_path: str,
    block_size: int = 1024,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    # Match the official evaluator: uint16 .bin, non-overlapping windows.
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    data = torch.from_numpy(data.astype(np.int64))

    total_loss = 0.0
    total_tokens = 0
    n_chunks = (len(data) - 1) // block_size
    n_chunks = (n_chunks // batch_size) * batch_size

    if n_chunks == 0:
        raise ValueError(
            f"Data too small: {len(data)} tokens. Need at least {block_size * batch_size + 1}."
        )

    for i in range(0, n_chunks, batch_size):
        input_ids = torch.stack(
            [data[j * block_size : j * block_size + block_size] for j in range(i, i + batch_size)]
        ).to(device)
        targets = torch.stack(
            [data[j * block_size + 1 : j * block_size + block_size + 1] for j in range(i, i + batch_size)]
        ).to(device)

        logits = model(input_ids).logits
        if logits.shape[-1] != 50257:
            raise ValueError(f"Expected vocab size 50257, got {logits.shape[-1]}")

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return {
        "perplexity": perplexity,
        "avg_loss_nats": avg_loss,
        "total_tokens_evaluated": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate gpt2-xl on official val.bin")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-xl",
        help="Hugging Face model name or local path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/data/fengfei/cse251b-nanogpt/val.bin",
        help="Path to tokenized uint16 val/test .bin file",
    )
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.is_bf16_supported() else None

    print(f"Loading teacher model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher params: {total_params:,}")
    print(f"Teacher vocab_size: {model.config.vocab_size}")
    print(f"Evaluating on {args.data} (block_size={args.block_size}, batch_size={args.batch_size})")

    t0 = time.time()
    results = compute_perplexity(
        model=model,
        data_path=args.data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    elapsed = time.time() - t0

    print("=" * 50)
    print(f"Model:        {args.model_name}")
    print(f"Perplexity:   {results['perplexity']:.4f}")
    print(f"Avg loss:     {results['avg_loss_nats']:.6f}")
    print(f"Tokens eval:  {results['total_tokens_evaluated']}")
    print(f"Eval time:    {elapsed:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
