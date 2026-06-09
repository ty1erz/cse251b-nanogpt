# Model card — try

**Generated:** 2026-05-31T18:19:13

## Architecture
- Class: `model_v5.GPT` ([model_v5.py](../../model_v5.py))
- Config: `n_layer=20`, `n_head=9`, `n_kv_head=3`, `n_embd=576`, `mlp_hidden=1536`, `block_size=1024`, `vocab_size=50304`
- **Parameters:** 99,780,160  (99.78 M)
- Components: RoPE, RMSNorm, **SwiGLU MLP + grouped-query attention**, QK-Norm, tied embeddings, bias-free linears

## Training data (v5 mix)
- Mix: fineweb=50%, wikipedia=20%, science=15%, books=15%
- Loader: `MixedDataLoader` ([mix_loader.py](../../mix_loader.py))

## Optimizer
- **Muon** for 70,778,880 2-D hidden parameters
  - lr=0.013, momentum=0.95, ns_steps=5, nesterov=True, wd=0.1
- **AdamW** for 29,001,280 embedding + 1-D parameters
  - lr=0.00052, betas=(0.9, 0.95), wd_embed=0, wd_norm=0
- Both schedulers: warmup 200 → cosine to 0.1× peak

## Hyperparameters
- micro_batch=16, seq_len=1024, total_batch_size=524288
- max_steps=38000 (≈ 19.92 B tokens)
- ckpt_every=2000, eval_every=250
- grad_clip=1.0

## Results
- Final step: 37999
- **Final val loss (mix val shard):** 3.0553
- val.bin perplexity: (not run; val.bin not found)

## Reproducibility
- Started: 2026-05-30T03:52:09
- Ended:   2026-05-31T18:19:13
- Wall-clock: 38.45 h
- Seed: 1337
- Run script: `train_v5.py --run_name try --max_steps 38000`

Checkpoints in this directory: every 2000 steps + final.
