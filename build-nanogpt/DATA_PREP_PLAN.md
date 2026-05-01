# High-Quality Data Preparation Plan (Sub-100M GPT)

## Why this plan
For a sub-100M model, data quality and distribution match usually matter more than brute-force data scale. This plan emphasizes curated, clean, high-information documents and careful train/val separation.

## Recommended sources and default ratios
If course/target-domain data is available:
- 50% target-domain/course-provided training corpus
- 30% FineWeb-Edu (`HuggingFaceFW/fineweb-edu`)
- 15% Cosmopedia textbook/article-style (`HuggingFaceTB/cosmopedia`)
- 5% OpenWebMath (`open-web-math/open-web-math`) when useful

If no target-domain data is available:
- 60% FineWeb-Edu
- 30% Cosmopedia
- 10% OpenWebMath or clean Wikipedia/book-style text

## Pipeline implemented in `prepare_pretrain_data.py`
1. Load each dataset from Hugging Face or local files (JSONL/TXT directory).
2. Sample each dataset by configurable ratio and caps:
   - document cap: `--max_documents_per_dataset`
   - approximate token budget: `--target_num_tokens`
3. Normalize and filter at document level:
   - remove empty docs
   - remove very short docs (`--min_doc_chars`)
   - remove very long docs (`--max_doc_chars`)
   - normalize whitespace; keep punctuation/casing/symbols
4. Deduplicate:
   - exact dedupe (global hash set)
   - optional approximate near-dedupe (`--near_dedup`) via SimHash buckets
5. Shuffle documents.
6. Split train/val at document level (`--val_fraction`) to reduce leakage.
7. Train a custom BPE tokenizer on the mixture (`--tokenizer_vocab_size`, e.g., 8000/16000).
8. Tokenize and write nanoGPT-compatible `train.bin` and `val.bin` (uint16).
9. Save `metadata.json` with dataset stats, ratios, estimated tokens, and outputs.

## Config files
- Smoke config (default-safe):
  - `data_configs/mix_smoke_16k.json`
- Full templates:
  - `data_configs/mix_full_with_target_16k.template.json`
  - `data_configs/mix_full_no_target_16k.template.json`

## Quick start (smoke mode)
```bash
python prepare_pretrain_data.py \
  --mode smoke \
  --dataset_mix_config data_configs/mix_smoke_16k.json \
  --target_num_tokens 500000 \
  --tokenizer_vocab_size 16000 \
  --output_dir prepared_mixture_smoke_16k
```

## GPT-2 tokenizer mode (vocab 50257)
When you need strict README-compatible vocabulary:

```bash
python prepare_pretrain_data.py \
  --mode smoke \
  --dataset_mix_config data_configs/mix_smoke_16k.json \
  --target_num_tokens 500000 \
  --tokenizer_backend gpt2_tiktoken \
  --output_dir prepared_mixture_gpt2
```

## Full run with target-domain anchor
1. Edit local path in `mix_full_with_target_16k.template.json`.
2. Run:

```bash
python prepare_pretrain_data.py \
  --mode full \
  --dataset_mix_config data_configs/mix_full_with_target_16k.template.json \
  --target_num_tokens 50000000 \
  --max_documents_per_dataset 200000 \
  --tokenizer_vocab_size 16000 \
  --output_dir prepared_mixture_full_16k \
  --near_dedup
```

## Notes for ablations
- Tokenizer size: run the same config with `--tokenizer_vocab_size 8000` and `16000`.
- Dataset mixture: adjust ratios in config JSON only; keep model/training fixed.
- Domain matching: increase target-domain ratio when validation distribution is in-domain.
- OpenWebMath: keep small unless validation is STEM-heavy.
