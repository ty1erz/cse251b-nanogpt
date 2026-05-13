# Model card — v3_new

**Generated:** 2026-05-02T19:02:57

## Architecture
- Class: `model_v2.GPT` ([model_v2.py](../../model_v2.py))
- Config: `n_layer=12`, `n_head=10`, `n_embd=640`, `block_size=1024`, `vocab_size=50304`
- **Parameters:** 91,194,496  (91.19 M)
- Modern components: RoPE, RMSNorm, ReLU² MLP, QK-Norm, tied embeddings, bias-free linears

## Training data (v3 mix — FineWeb-heavy)
- Mix: fineweb=50%, wikipedia=20%, science=15%, books=15%
- Loader: `MixedDataLoader` ([mix_loader.py](../../mix_loader.py))

## Optimizer (Phase 2)
- **Muon** for 58,982,400 2-D hidden parameters
  - lr=0.02, momentum=0.95, ns_steps=5, nesterov=True, wd=0.1
- **AdamW** for 32,212,096 embedding + 1-D parameters
  - lr=0.0008, betas=(0.9, 0.95), wd_embed=0, wd_norm=0
- Both schedulers: warmup 200 → cosine to 0.1× peak

## Hyperparameters
- micro_batch=16, seq_len=1024, total_batch_size=524288
- max_steps=5100 (≈ 2.67 B tokens)
- ckpt_every=1700, eval_every=250
- grad_clip=1.0

## Results
- Final step: 5099
- **Final val loss (FineWeb-Edu val shard):** 3.2531
- **val.bin perplexity:** 25.9094
- val.bin avg loss (nats): 3.2546
- tokens evaluated: 5,169,152

## Reproducibility
- Started: 2026-05-02T12:32:29
- Ended:   2026-05-02T19:02:57
- Wall-clock: 6.51 h
- Seed: 1337
- Run script: `train_v3.py --run_name v3_new --max_steps 5100`

Checkpoints in this directory: every 1700 steps + final.
