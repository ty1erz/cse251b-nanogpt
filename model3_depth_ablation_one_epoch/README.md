# Model 3: Depth Ablation + Low-LR Continuation

Model 3 is the deepest, thinnest member of the architecture family: a 20-layer,
576-dimensional decoder with aggressive 1:3 grouped-query attention. It is the
**strongest architecture at the matched 38k budget** — 20.918 public-validation
PPL, beating Model 2 by 0.245 and Model 1 by 0.407 (report Table 6).

The final report noted that Model 3 *stopped after the 38k depth-ablation run*
and was never given the low-learning-rate continuation that Models 1 and 2
received, so a fair architecture ranking was left open. **This folder closes
that gap**: it adds the continuation stage and the val.bin-driven checkpoint
selection used by the rest of the family.

## Contents

- `model.py`: the exact 20L/576d architecture, plus the evaluator entrypoint
  (`load_model` / `EvalGPT`).
- `train_model3_depth_ablation_one_epoch.py`: the 38k main run (one matched epoch).
- `train_model3_low_lr_finetune.py`: the step-wise low-LR continuation with
  val.bin checkpoint selection (new).
- `config.json`: main-run architecture and hyperparameters.
- `config_finetune.json`: the continuation staircase schedule.
- `evaluate.py`: the unmodified course perplexity evaluator.
- `training_logs/`: text logs for the 38k main run and the continuation stages
  (checkpoints and token shards are intentionally excluded).

## Architecture

The model has 99,780,160 parameters (99.78M):

| Component | Configuration |
| --- | --- |
| Vocabulary / context | 50,304 padded (sliced to 50,257 for eval) / 1,024 tokens |
| Transformer | 20 decoder blocks, hidden size 576 |
| Attention | GQA: 9 query heads, 3 key/value heads, head size 64 |
| Position / normalization | RoPE / RMSNorm, with QK-Norm on queries and keys |
| MLP | SwiGLU, intermediate size 1,536 |
| Embeddings | Token embedding tied to the LM head |
| Bias / dropout | Disabled / 0.0 |

Run `python model.py` to print the parameter count and a forward-pass shape check.

The 1:3 GQA layout (3 KV heads for 9 query heads) costs roughly `8/3 d^2` per
layer in attention projections versus `4 d^2` for standard MHA. Those savings are
reinvested into depth — 20 layers versus Model 1's 13 — under the same sub-100M cap.

## Environment

```bash
pip install -r ../requirements.txt
```

The shared `common/mix_loader.py` loader expects the four tokenized sources
(FineWeb-Edu, Wikipedia, science, books) as uint16 shards. Point it at the
machine's data root and supply the public validation file:

```bash
export NANOGPT_DATA_ROOT=/path/to/data_root   # contains edu_fineweb10B/, data/wikipedia/, ...
export VAL_BIN=/path/to/val.bin               # defaults to <experiment_root>/val.bin
```

The data mixture is FineWeb-Edu 50%, Wikipedia 20%, science 15%, books 15%.
Training uses sequence length 1,024 and a global batch of 524,288 tokens.

## Main run (38k depth ablation)

One matched epoch: 38,000 updates over ~19.92B sampled tokens, Muon for the
2-D hidden matrices and AdamW for embeddings and 1-D parameters, warmup 200 then
cosine decay to 10% of peak.

- Peak Muon LR: `1.3e-2`
- Peak AdamW LR: `5.2e-4`
- Warmup: 200 steps; cosine decay over 38,000 steps; min LR ratio 0.1
- **Public-validation PPL at 38k: 20.918** (report Table 6)

```bash
python train_model3_depth_ablation_one_epoch.py \
  --run_name model3_depth_20L_one_epoch
```

The resulting resume checkpoint is `logs/model3_depth_20L_one_epoch/model_037999.pt`.

## Low-LR continuation (step-wise checkpoint search)

`train_model3_low_lr_finetune.py` continues the 38k checkpoint with a
**learning-rate staircase**. Each stage trains for a fixed number of updates at
a **constant** LR (no within-stage decay); during the stage, public `val.bin`
perplexity is evaluated every 250 steps and the lowest-PPL checkpoint is mirrored
to `best.pt`. The next stage resumes from that `best.pt` at a lower LR.

This is the report's "continuation as local search": each branch starts from a
strong saved state and lowers the schedule scale, while the architecture, data
mixture, objective, optimizer split, and loader position are all preserved.
Unlike the main run, **model selection uses the official `val.bin` metric, not
the internal mixed-shard validation loss**.

Recommended staircase (38k → 43k in five 1,000-step stages):

| Stage | Resume from | → step | Muon LR | AdamW LR |
| ---: | --- | ---: | ---: | ---: |
| 1 | `model_037999.pt` (main run) | 39,000 | `8e-4` | `3.2e-5` |
| 2 | `model3_ft_s1/best.pt` | 40,000 | `4e-4` | `1.6e-5` |
| 3 | `model3_ft_s2/best.pt` | 41,000 | `2e-4` | `8e-6` |
| 4 | `model3_ft_s3/best.pt` | 42,000 | `1e-4` | `4e-6` |
| 5 | `model3_ft_s4/best.pt` | 43,000 | `4e-5` | `2.5e-6` |

```bash
# stage 1 (defaults already encode this stage)
python train_model3_low_lr_finetune.py \
  --run_name model3_ft_s1 \
  --resume logs/model3_depth_20L_one_epoch/model_037999.pt \
  --max_steps 39000 --muon_lr 8e-4 --adam_lr 3.2e-5

# stage 2 (repeat the pattern, lowering the LR each time)
python train_model3_low_lr_finetune.py \
  --run_name model3_ft_s2 \
  --resume logs/model3_ft_s1/best.pt \
  --max_steps 40000 --muon_lr 4e-4 --adam_lr 1.6e-5
```

The remaining stages (3–5) follow the table; the full command list is in the
docstring at the top of `train_model3_low_lr_finetune.py`. The submission
checkpoint is the lowest-PPL stage's `best.pt`.

### Observed results

Measured `val.bin` perplexity for the continuation of a local 38k checkpoint.
Each row is the best checkpoint of its stage; "step" is where the minimum
occurred, which is often before the stage's nominal end (the staircase keeps the
best, not the last, checkpoint):

| Stage | Constant Muon / AdamW LR | Best step | val.bin PPL |
| ---: | --- | ---: | ---: |
| main run | (cosine to `1.3e-3`) | 38,000 | 20.918 |
| 1 | `8e-4` / `3.2e-5` | 38,500 | 20.068 |
| 2 | `4e-4` / `1.6e-5` | 39,000 | 19.569 |
| 3 | `2e-4` / `8e-6` | 39,999 | 19.271 |
| 4 | `1e-4` / `4e-6` | — | *(continue per recipe)* |
| 5 | `4e-5` / `2.5e-6` | — | *(continue per recipe)* |

The first two stages already recover ~1.35 PPL, consistent with the report's
finding that the earliest continuation updates deliver the largest return. The
ppl can rise within a stage once the constant LR is too high for the current
state (stage 1 bottomed at step 38,500 and drifted up by 38,999), which is
exactly why each stage keeps its `best.pt` and the next stage branches from it.

## Evaluation

`load_model(checkpoint_path, device)` in `model.py` rebuilds the network and
returns an `EvalGPT` whose `forward(input_ids)` yields logits over the 50,257
evaluator classes. A submission directory needs `model.py` and a checkpoint:

```bash
python evaluate.py \
  --model_dir /path/to/submission_dir \
  --checkpoint_filename best.pt \
  --data "$VAL_BIN" \
  --device cuda
```

`evaluate.py` reports perplexity, mean loss, token count, and parameter count
(it warns if the model exceeds the 100M cap). Every full pass covers 5,169,152
target tokens.
