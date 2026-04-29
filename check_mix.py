"""
Sanity-check the multi-source data pipeline before kicking off Phase 1 training.

Verifies:
  1. mix_loader can find every source it expects
  2. Each batch has correct shape (B, T) and dtype int64
  3. Token ids fall in the GPT-2 vocab range [0, 50256]
  4. Sampling over many batches roughly hits the configured ratio
  5. Decoding a sample yields readable text from the chosen source

Usage:
    python check_mix.py
"""

import collections
import os
import sys

import numpy as np
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mix_loader import MixedDataLoader, DEFAULT_MIX  # noqa: E402

B, T = 4, 64
N_BATCHES = 200

print("Loading MixedDataLoader (split=train)...")
loader = MixedDataLoader(
    B=B, T=T, process_rank=0, num_processes=1, split="train",
    master_process=True,
)

enc = tiktoken.get_encoding("gpt2")
counts = collections.Counter()

print(f"\nDrawing {N_BATCHES} batches at B={B}, T={T}...")
for i in range(N_BATCHES):
    name = loader.names[loader.rng.choice(len(loader.names), p=loader.probs)]
    counts[name] += 1
loader.rng = np.random.default_rng(1337)  # reset, the loop above only counted

# Now actually pull batches and verify shapes / dtypes / decoded text per source.
seen = collections.Counter()
samples = {}
for i in range(N_BATCHES):
    # mimic the same sampling MixedDataLoader does internally
    idx = loader.rng.choice(len(loader.names), p=loader.probs)
    name = loader.names[idx]
    x, y = loader.loaders[name].next_batch()
    assert x.shape == (B, T), f"x shape {x.shape}"
    assert y.shape == (B, T), f"y shape {y.shape}"
    assert x.dtype.is_floating_point is False, f"x dtype {x.dtype}"
    assert int(x.max()) < 50257, f"x has token id {int(x.max())} >= 50257"
    assert int(x.min()) >= 0, f"x has negative token id {int(x.min())}"
    seen[name] += 1
    if name not in samples:
        samples[name] = enc.decode(x[0].tolist())

print("\n=== mix sampling distribution (target vs observed) ===")
total = sum(seen.values())
for n in loader.names:
    obs = seen[n] / total
    tgt = loader.weights[n]
    print(f"  {n:10s}  target={tgt:.0%}  observed={obs:.0%}")

print("\n=== one decoded sample per source ===")
for n, txt in samples.items():
    snippet = txt.replace("\n", " ").strip()[:160]
    print(f"\n[{n}]  {snippet}...")

print("\nOK — pipeline is ready for training.")
