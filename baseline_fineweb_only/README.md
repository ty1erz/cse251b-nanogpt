# GPT Baseline Data-Source Experiments

All three runs use the same 8-layer, 512-dimensional GPT architecture and the
same 5,000-step AdamW training schedule. Only the training data changes.

| Trainer | Training data | Validation perplexity |
| --- | --- | ---: |
| `train_baseline_fineweb_only.py` | FineWeb-Edu only | 39.67 |
| `../baseline_openwebtext_only/train_baseline_openwebtext_only.py` | OpenWebText only | 65.21 |
| `../baseline_mix_50_20_15_15/train_baseline_mix_50_20_15_15.py` | FineWeb/Wikipedia/science/books, 50/20/15/15 | **38.8281** |

Run the single-source alternatives from the repository root:

```bash
python baseline_fineweb_only/train_baseline_fineweb_only.py
python baseline_openwebtext_only/train_baseline_openwebtext_only.py
```

The expected token-file layouts and data path overrides are documented in the
repository-level `README.md`.
