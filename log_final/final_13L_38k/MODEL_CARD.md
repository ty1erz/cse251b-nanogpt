# Model card — final_13L_38k

**Generated:** 2026-05-12T20:38:06

## Architecture
- Class: `model_v2.GPT` ([model_v2.py](../../model_v2.py))
- Config: `n_layer=13`, `n_head=10`, `n_embd=640`, `block_size=1024`, `vocab_size=50304`
- **Parameters:** 96,111,104  (96.11 M)
- Modern components: RoPE, RMSNorm, ReLU² MLP, QK-Norm, tied embeddings, bias-free linears

## Training data (final mix)
- Mix: fineweb=50%, wikipedia=20%, science=15%, books=15%
- Loader: `MixedDataLoader` ([mix_loader.py](../../mix_loader.py))

## Optimizer
- **Muon** for 63,897,600 2-D hidden parameters
  - lr=0.02, momentum=0.95, ns_steps=5, nesterov=True, wd=0.1
- **AdamW** for 32,213,504 embedding + 1-D parameters
  - lr=0.0008, betas=(0.9, 0.95), wd_embed=0, wd_norm=0
- Both schedulers: warmup 200 → cosine to 0.1× peak

## Hyperparameters
- micro_batch=16, seq_len=1024, total_batch_size=524288
- max_steps=38000 (≈ 19.92 B tokens)
- ckpt_every=2000, eval_every=250
- grad_clip=1.0

## Results
- Final step: 37999
- **Final val loss (mix val shard):** 3.1096
- **val.bin perplexity:** 22.1834
- val.bin avg loss (nats): 3.0993
- tokens evaluated: 5,169,152

## Reproducibility
- Started: 2026-05-12T09:45:54
- Ended:   2026-05-12T20:38:06
- Wall-clock: 10.87 h
- Seed: 1337
- Run script: `train_final.py --run_name final_13L_38k --max_steps 38000`

Checkpoints in this directory: every 2000 steps + final.
