# Model card — v4_full_5100

**Generated:** 2026-04-26T22:47:27

## Architecture (Phase 1 + Phase 3 softcap)
- Class: `model_v2.GPT` ([model_v2.py](../../model_v2.py))
- Config: `n_layer=13`, `n_head=10`, `n_embd=640`, `block_size=1024`, `vocab_size=50304`
- **Parameters:** 96,111,104  (96.11 M)
- Modern components: RoPE, RMSNorm, ReLU² MLP, QK-Norm, tied embeddings, bias-free linears
- **Logit softcap:** cap = 30.0  (Gemma-2 style: `cap * tanh(logits/cap)`)

## Training data (v4 mix — same as v3)
- Mix: fineweb=51%, wikipedia=20%, science=15%, books=15%

## Optimizer (Phase 2 — Muon + AdamW split)
- **Muon** for 63,897,600 2-D hidden parameters
  - lr=0.02, momentum=0.95, ns_steps=5, nesterov=True, wd=0.1
- **AdamW** for 32,213,504 embedding + 1-D parameters
  - lr=0.0008, betas=(0.9, 0.95), wd_embed=0, wd_norm=0
- Both schedulers: warmup 200 → cosine to 0.1× peak

## Auxiliary losses (Phase 3)
- **MTP +2** (shared head, no extra params): weight = 0.3
- **Z-loss** (`(logsumexp(logits))²` regularizer): weight = 0.0001
- **Logit softcap** (in-model): cap = 30.0

## Speedup
- `torch.compile`: disabled

## Hyperparameters
- micro_batch=16, seq_len=1024, total_batch_size=524288
- max_steps=5100 (≈ 2.67 B tokens)
- ckpt_every=1700, eval_every=250
- grad_clip=1.0

## Results
- Final step: 5099
- **Final val loss (FineWeb-Edu val shard):** inf
- **val.bin perplexity:** 30.3505
- val.bin avg loss (nats): 3.4128
- tokens evaluated: 5,169,152

## Reproducibility
- Started: 2026-04-26T22:47:27
- Ended:   2026-04-26T22:47:27
- Wall-clock: 0.00 h
- Seed: 1337
- Run script: `train_v4.py --run_name v4_full_5100 --max_steps 5100`

Checkpoints in this directory: every 1700 steps + final.
