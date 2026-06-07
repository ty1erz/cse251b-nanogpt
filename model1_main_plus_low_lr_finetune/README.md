# Model 1 Data-Mixture Experiments

All three runs use the same 13-layer, 640-dimensional Model 1 architecture and
the same 38,000-step training schedule. Only the source mixture changes.

| Trainer | FineWeb | Wikipedia | Science | Books | 38k validation PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_model1_mix50.py` | 50% | 20% | 15% | 15% | **21.3251** |
| `train_model1_mix53.py` | 53% | 19% | 14% | 14% | 21.4597 |
| `train_model1_mix56_main_plus_low_lr_finetune.py` | 56% | 18% | 13% | 13% | 21.6089 |

The 50% and 53% scripts reproduce the one-epoch ratio ablations run on the
other pods. The 56% script is the experiment that was subsequently continued
with progressively lower learning rates.

Run the one-epoch alternatives from the repository root:

```bash
python model1_main_plus_low_lr_finetune/train_model1_mix50.py \
  --run_name model1_mix50_main_38k

python model1_main_plus_low_lr_finetune/train_model1_mix53.py \
  --run_name model1_mix53_main_38k
```

Run the 56% main stage:

```bash
python model1_main_plus_low_lr_finetune/train_model1_mix56_main_plus_low_lr_finetune.py \
  --run_name model1_mix56_main_38k
```

For low-LR continuation, resume the 56% checkpoint with the staged learning
rates documented in the repository-level `README.md`.
