# Final Experiment Bundles

This directory contains cleaned copies of the four final experiment families.
Checkpoints and large token shards are intentionally not duplicated.

## Data layout

By default, the shared loader expects:

```text
<project-root>/edu_fineweb10B/
<project-root>/data/wikipedia/
<project-root>/data/science/
<project-root>/data/books/
```

For RunPod or another machine, point to the directory containing those paths:

```bash
export NANOGPT_DATA_ROOT=/workspace/build-nanogpt
```

The public validation file defaults to
`<project-root>/cse251b-nanogpt/val.bin`. Override it with
`--val_bin_path /path/to/val.bin` when necessary.

## 1. Model 1: main run and low-LR fine-tuning

Directory: `model1_main_plus_low_lr_finetune/`

- 13 layers, hidden size 640, 10 attention heads
- RoPE, RMSNorm, QK-Norm, SwiGLU
- 96.64M parameters
- Data mix: 56/18/13/13
- Main stage: 38,000 steps, approximately 19.9B tokens
- Second stage: resume from checkpoints with progressively lower learning rates

Main run:

```bash
python model1_main_plus_low_lr_finetune/train_model1_main_plus_low_lr_finetune.py \
  --run_name model1_main_38k
```

First continuation stage:

```bash
python model1_main_plus_low_lr_finetune/train_model1_main_plus_low_lr_finetune.py \
  --run_name model1_low_lr_to_42k \
  --resume model1_main_plus_low_lr_finetune/logs/model1_main_38k/model_037999.pt \
  --max_steps 42000 \
  --muon_lr 6e-3 \
  --adam_lr 2.4e-4 \
  --min_lr_ratio 0.1 \
  --ckpt_every 500
```

Continue the same pattern from the selected checkpoint, reducing the Muon and
AdamW learning rates. The original sequence used approximate peak LR pairs:
`(6e-3, 2.4e-4)`, `(4e-3, 1.8e-4)`, `(3e-3, 1.35e-4)`,
`(1.5e-3, 8e-5)`, `(1.2e-3, 6.4e-5)`, `(9e-4, 5e-5)`,
`(7e-4, 4e-5)`, and `(4e-4, 2.5e-5)`.

## 2. Model 3: depth ablation

Directory: `model3_depth_ablation_one_epoch/`

- 20 layers, hidden size 576
- 9 query heads and 3 KV heads
- RoPE, RMSNorm, QK-Norm, SwiGLU, GQA
- Approximately 99.78M parameters
- Data mix: 50/20/15/15
- One 38,000-step main run only

```bash
python model3_depth_ablation_one_epoch/train_model3_depth_ablation_one_epoch.py \
  --run_name model3_depth_20L_one_epoch
```

## 3. GPT baseline: mixed-data ratio experiment

Directory: `baseline_mix_50_20_15_15/`

- 8 layers, hidden size 512, 8 heads
- Learned positional embeddings, LayerNorm, GELU MLP
- AdamW optimizer
- Data mix fixed to 50/20/15/15
- 5,000 training steps

```bash
python baseline_mix_50_20_15_15/train_baseline_mix_50_20_15_15.py
```

## 4. Model 2: two-epoch run

Directory: `model2_two_epoch_run/`

This bundle is merged from the `alvin_clean` branch and retains that branch in
the Git history.

- 16 layers, hidden size 640
- 10 query heads and 5 KV heads
- RoPE, RMSNorm, SwiGLU, and GQA
- 99.03M parameters
- Data mix: 50/20/15/15
- First epoch: 38,000 steps
- Second epoch: low-LR continuation to step 71,800
- Best recorded validation perplexity: 18.7681 at step 69,700

See `model2_two_epoch_run/README.md` for the exact staged learning rates,
environment variables, launch commands, and evaluation instructions.

## Shared files

- `common/mix_loader.py`: mixed-domain shard loader
- `common/muon.py`: Muon optimizer and parameter grouping
- Each experiment includes `config.json`.
- Model 1 and Model 3 include their training architecture as `model.py`.
- The baseline includes an evaluator-compatible `model.py`; its trainer retains
  the original inline architecture definition.
