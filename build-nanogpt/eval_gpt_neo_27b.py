#!/usr/bin/env python3
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


EXPECTED_VOCAB_SIZE = 50257


@torch.no_grad()
def compute_perplexity(model, data_path, block_size, batch_size, device):
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
        if logits.shape[-1] != EXPECTED_VOCAB_SIZE:
            raise ValueError(f"Expected vocab size {EXPECTED_VOCAB_SIZE}, got {logits.shape[-1]}")

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    return {
        "perplexity": math.exp(avg_loss),
        "avg_loss_nats": avg_loss,
        "total_tokens_evaluated": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-Neo 2.7B on the official val.bin")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--data", type=str, default="/data/fengfei/cse251b-nanogpt/val.bin")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--check_vocab_only", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    print(f"Checking config: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    print(f"Config vocab_size: {config.vocab_size}")
    if config.vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"{args.model_name} vocab_size={config.vocab_size}, expected {EXPECTED_VOCAB_SIZE}."
        )
    print("Vocabulary matches student: yes")

    if args.check_vocab_only:
        return

    dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.is_bf16_supported() else None
    print(f"Loading model: {args.model_name}")
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).to(args.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")
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
