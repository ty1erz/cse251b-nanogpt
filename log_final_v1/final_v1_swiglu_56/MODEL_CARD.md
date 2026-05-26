# Model card — final_v1_swiglu_56

**Generated:** 2026-05-25T04:03:12

## Architecture
- Class: `model_final.GPT` ([model_final.py](../../model_final.py))
- Config: `n_layer=13`, `n_head=10`, `n_embd=640`, `mlp_hidden=1728`, `block_size=1024`, `vocab_size=50304`
- **Parameters:** 96,643,584  (96.64 M)
- Components: RoPE, RMSNorm, **SwiGLU MLP**, QK-Norm, tied embeddings, bias-free linears

## Training data (final_v1 mix)
- Mix: fineweb=56%, wikipedia=18%, science=13%, books=13%
- Loader: `MixedDataLoader` ([mix_loader.py](../../mix_loader.py))

## Optimizer
- **Muon** for 64,430,080 2-D hidden parameters
  - lr=0.013, momentum=0.95, ns_steps=5, nesterov=True, wd=0.1
- **AdamW** for 32,213,504 embedding + 1-D parameters
  - lr=0.00052, betas=(0.9, 0.95), wd_embed=0, wd_norm=0
- Both schedulers: warmup 200 → cosine to 0.1× peak

## Hyperparameters
- micro_batch=16, seq_len=1024, total_batch_size=524288
- max_steps=38000 (≈ 19.92 B tokens)
- ckpt_every=2000, eval_every=250
- grad_clip=1.0

## Results
- Final step: 37999
- **Final val loss (mix val shard):** 3.0731
- val.bin perplexity: (not run; val.bin not found)

## Reproducibility
- Started: 2026-05-23T07:17:34
- Ended:   2026-05-25T04:03:12
- Wall-clock: 44.76 h
- Seed: 1337
- Run script: `train_final_v1.py --run_name final_v1_swiglu_56 --max_steps 38000`

Checkpoints in this directory: every 2000 steps + final.
