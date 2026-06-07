# CSE 251B NanoGPT: Two-Epoch Training

This branch contains only the code needed to reproduce the v13/v16 training
path:

- `train_first_epoch.py`: renamed v13 code, trained from scratch to step 38,000.
- `train_second_epoch.py`: renamed v16 code, continued from step 38,000.
- `model.py`: the exact architecture used by both stages and by evaluation.
- `muon.py`: Muon optimizer implementation.
- `evaluate.py`: perplexity evaluation script.

Checkpoints, tokenized data, generated submissions, and raw logs are intentionally
excluded from this clean branch.

## Model Architecture

The model has 99,032,320 parameters:

| Component | Configuration |
| --- | --- |
| Vocabulary / context | 50,257 tokens / 1,024 tokens |
| Transformer | 16 decoder blocks, hidden size 640 |
| Attention | GQA: 10 query heads, 5 key/value heads, head size 64 |
| Position / normalization | RoPE / RMSNorm |
| MLP | SwiGLU, intermediate size 1,536 |
| Embeddings | Token embedding tied to the LM head |
| Bias / dropout | Disabled / 0.0 |

Run `python model.py` to print the module tree and parameter count.

## Environment

```bash
pip install -r requirements.txt
```

The loaders accept either a directory containing `train.bin`, a directory of
`.npy` shards, or a tokenized binary file. Override the default machine paths
with environment variables:

```bash
export FINEWEB_PATH=/path/to/fineweb
export WIKIPEDIA_PATH=/path/to/wikipedia
export SCIENCE_PATH=/path/to/science
export BOOKS_PATH=/path/to/books
export VAL_BIN=/path/to/val.bin
```

The data mixture is FineWeb 50%, Wikipedia 20%, science 15%, and books 15%.
Training uses sequence length 1,024 and a global batch of 524,288 tokens.

## First Epoch

The first stage used hard cross-entropy only, with no teacher or distillation.
It ran 38,000 optimizer steps, approximately 19.9B training tokens, and took
about 4.5 to 5 days on 8 GPUs.

- Muon peak LR: `1.3e-2`
- AdamW peak LR: `5.2e-4`
- Warmup: 200 steps
- Cosine decay: 38,000 steps
- Minimum LR ratio: 0.1
- Final validation perplexity at step 38,000: 21.1630

```bash
torchrun --standalone --nproc_per_node=8 train_first_epoch.py \
  --target_step 38000
```

The resulting resume checkpoint is
`log/first_epoch/train_ckpt_38000.pt`.

## Second Epoch

The second stage continued from step 38,000 to the exported step-71,800 model:
33,800 additional optimizer steps, approximately 17.7B tokens, and about 4 days
on 8 GPUs. The original final target was step 72,000; the recorded export used
for evaluation was step 71,800.

The peak learning rate was reduced manually four times:

| Global steps | Muon peak LR | AdamW peak LR |
| --- | ---: | ---: |
| 38,000 to 54,000 | `4e-4` | `1.6e-5` |
| 54,000 to 62,000 | `2e-4` | `8e-6` |
| 62,000 to 63,600 | `1.5e-4` | `6e-6` |
| 63,600 to 69,600 | `1e-4` | `4e-6` |
| 69,600 to 71,800 | `6e-5` | `2.4e-6` |

Each command uses the global cosine schedule with a 0.5 minimum LR ratio.
Because the schedule is evaluated at the global step, the effective logged LR
is lower than the peak value passed on the command line.

```bash
torchrun --standalone --nproc_per_node=8 train_second_epoch.py \
  --resume log/first_epoch/train_ckpt_38000.pt \
  --target_step 54000 --lr_decay_steps 76000 \
  --muon_lr 4e-4 --adam_lr 1.6e-5 --min_lr_ratio 0.5 \
  --save_interval 500 --eval_interval 500 \
  --global_lr_schedule --run_name second_epoch_lr4e4

torchrun --standalone --nproc_per_node=8 train_second_epoch.py \
  --resume log/second_epoch/train_ckpt_54000.pt \
  --target_step 62000 --lr_decay_steps 76000 \
  --muon_lr 2e-4 --adam_lr 8e-6 --min_lr_ratio 0.5 \
  --save_interval 500 --eval_interval 500 \
  --global_lr_schedule --run_name second_epoch_lr2e4

torchrun --standalone --nproc_per_node=8 train_second_epoch.py \
  --resume log/second_epoch/train_ckpt_62000.pt \
  --target_step 63600 --lr_decay_steps 64000 \
  --muon_lr 1.5e-4 --adam_lr 6e-6 --min_lr_ratio 0.5 \
  --save_interval 400 --eval_interval 100 \
  --global_lr_schedule --run_name second_epoch_lr1p5e4

torchrun --standalone --nproc_per_node=8 train_second_epoch.py \
  --resume log/second_epoch/train_ckpt_63600.pt \
  --target_step 69600 --lr_decay_steps 70000 \
  --muon_lr 1e-4 --adam_lr 4e-6 --min_lr_ratio 0.5 \
  --save_interval 200 --eval_interval 100 \
  --global_lr_schedule --run_name second_epoch_lr1e4

torchrun --standalone --nproc_per_node=8 train_second_epoch.py \
  --resume log/second_epoch/train_ckpt_69600.pt \
  --target_step 71800 --lr_decay_steps 72000 \
  --muon_lr 6e-5 --adam_lr 2.4e-6 --min_lr_ratio 0.5 \
  --save_interval 100 --eval_interval 50 \
  --global_lr_schedule --run_name second_epoch_lr6e5
```

The final recorded validation perplexity was 18.7994 at step 71,800. The best
recorded value in the final run was 18.7681 at step 69,700.

## Evaluation

Training exports a submission directory containing `model.py`, `config.json`,
and `checkpoint.pt`. Evaluate one with:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --model_dir log/second_epoch/submission_DATE_NAME_71800 \
  --data "$VAL_BIN" \
  --device cuda
```
