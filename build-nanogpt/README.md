# build-nanogpt Experiment README

This directory contains the main experiment scripts used for the CSE 251B NanoGPT competition.

The student model architecture used in the main line (`v1`, `v4`, `v8`, `v10`, `v11`, `v12`) is the same 99.03M-parameter GPT-style model unless explicitly noted otherwise:

- decoder-only GPT
- vocab size: `50257`
- context length: `1024`
- `n_layer=16`
- `n_head=10`
- `n_kv_head=5`
- `n_embd=640`
- `RMSNorm + RoPE + SwiGLU + GQA`

The official evaluation script is:

- `../evaluate.py`
- official validation set: `../val.bin`

The official metric reported throughout this README is **perplexity on `val.bin`**.

---

## 1. Data Layout

### 1.1 Core pretraining mixture

The original large training file is:

- `prepared_mixture_gpt2_full/train.bin`

It was built from two sources, concatenated in order:

1. `FineWeb-Edu sample-10BT`
2. `Cosmopedia web_samples_v2`

The exact token boundary is recorded in:

- `prepared_mixture_gpt2_full/metadata.json`

Current numbers:

- FineWeb tokens: `9,953,989,344`
- Cosmopedia tokens: `3,624,349,131`
- Total train tokens: `13,578,338,475`

This means:

- FineWeb occupies token range `[0, 9,953,989,344)`
- Cosmopedia occupies token range `[9,953,989,344, 13,578,338,475)`

For convenience, these were later split into standalone bins:

- `tokenized_sources/fineweb_full/train.bin`
- `tokenized_sources/cosmopedia_full/train.bin`

### 1.2 Additional standalone sources prepared in this repo

These were added later to make the late-stage distribution more similar to the official validation set:

- `tokenized_sources/openwebtext_2b/train.bin`
- `tokenized_sources/wikipedia_2b/train.bin`

### 1.3 External zzw-style shard sources

For the zzw-style mixture experiments, three external `.npy` shard directories are used from:

- `/data/fengfei/cse251b-nanogpt-zzv-train/data/wikipedia`
- `/data/fengfei/cse251b-nanogpt-zzv-train/data/science`
- `/data/fengfei/cse251b-nanogpt-zzv-train/data/books`

These are tokenized GPT-2 shard directories, not `train.bin` files.

### 1.4 Loader formats used in this repo

Two token source formats are used:

1. **single `train.bin` sources**
   - uint16 memmap
   - used by most `v*` scripts in this repo

2. **`.npy` shard directories**
   - used by zzw-style data prep
   - later supported in `v12` together with `train.bin`

---

## 2. Version History (`v*`)

Below is the practical summary of what each version tried.

### v1 — baseline long-run pretraining

File:

- `train_gpt2.py`

Main idea:

- train on the original FineWeb + Cosmopedia mixture
- no extra teacher
- standard AdamW training

Data:

- `prepared_mixture_gpt2_full/train.bin`
- sequentially consumes the combined file

Why it mattered:

- established the first strong baseline
- produced the checkpoints later reused by `v4`

---

### v2 — smaller effective batch / more update-efficient sampling

File:

- `train_gpt2_v2.py`

Main idea:

- keep the same architecture and same core data
- reduce effective batch pressure and increase optimizer updates per token
- random window sampling instead of simple sequential walk

Why it mattered:

- much more token-efficient than `v1`
- useful as an optimization ablation

---

### v3 — architecture-heavy experimental branch

File:

- `train_gpt2_v3.py`

Main ideas attempted:

- factorized embedding
- QK norm
- gated attention
- hybrid Muon + AdamW
- source-balanced range sampler

Outcome:

- experimentally interesting
- not competitive with the simpler strong lines
- not part of the final best path

---

### v4 — post-training continuation from strong `v1`

File:

- `train_gpt2_v4.py`

Main idea:

- resume from a strong `v1` checkpoint
- continue training on a more targeted late-stage mixture
- emphasized remaining FineWeb + Cosmopedia with a different schedule

Why it mattered:

- this was the first clearly strong post-training continuation stage
- it substantially improved over the raw `v1` continuation

---

### v5 — domain-shift continuation on the second source

File:

- `train_gpt2_v5.py`

Main idea:

- resume from `v4`
- continue on the second source only / strongly bias toward it

Why it mattered:

- tested whether shifting the late-stage distribution helps more than simply training longer

---

### v5_distill — conservative GPT-2 XL top-k KD

File:

- `train_gpt2_v5_distill.py`

Main idea:

- resume from a strong `v4` checkpoint
- keep hard CE as the main loss
- add conservative top-k GPT-2 XL distillation

Outcome:

- useful but modest gains
- showed that light KD can help, but aggressive KD is risky late in training

---

### v6 — hidden-state distillation

File:

- `train_gpt2_v6.py`

Main idea:

- distill hidden states from GPT-2 XL using projection heads and cosine loss

Outcome:

- hidden loss decreased, but official PPL got worse
- abandoned as a main direction

---

### v7 / v7_kd — teacher-guided data selection and selective KD

Files:

- `train_gpt2_v7.py`
- `train_gpt2_v7_kd.py`

Main ideas:

- use GPT-2 XL to score candidate windows
- select “high gap” windows where teacher outperforms student
- later try selective teacher-better KD

Outcome:

- interesting analysis tool
- not as effective as the simpler successful late-stage branches

---

### v8 — 4-source late-stage mixture + light GPT-2 XL KD

File:

- `train_gpt2_v8.py`

Data mixture:

- FineWeb remaining
- Cosmopedia
- OpenWebText
- Wikipedia

Typical weights during the successful runs:

- FineWeb remaining: `55%`
- Cosmopedia: `20%`
- OpenWebText: `15%`
- Wikipedia: `10%`

Main idea:

- use more validation-like data late in training
- keep CE as main objective
- add very light GPT-2 XL top-k KD

Why it mattered:

- this became the strongest mainline before `v12`
- it improved the model from the mid-24s into the low-24s and eventually to about `24.05`

Important checkpoint used later:

- `log/v8/submission_20260428_alvin_v8_30950`
- official PPL there: about `24.0495`

---

### v9 — teacher-guided hard-negative margin loss

File:

- `train_gpt2_v9.py`

Main idea:

- use teacher top-k negatives to create an auxiliary logit margin loss on top of CE

Status:

- experimental branch
- not the main best line

---

### v10 — 4-source pure CE continuation

File:

- `train_gpt2_v10.py`

Main idea:

- same 4-source late-stage mixture family as `v8`
- remove the teacher entirely
- pure hard CE continuation from `v4`

Outcome:

- positive, but weaker than `v8`
- useful as a no-teacher control

---

### v11 — equal-weight 4-source pure CE

File:

- `train_gpt2_v11.py`

Main idea:

- same general continuation framework as `v10`
- make all four sources equal: `25/25/25/25`
- raise the learning rate back toward a stronger continuation regime

Outcome:

- exploratory branch
- intended as a clean high-learning-rate comparison against `v10`

---

### v12 — zzw-style data mixture in the main repo (current best line)

File:

- `train_gpt2_v12.py`

Main idea:

- keep the main repo student architecture and submission/eval flow
- resume from the strong `v8` checkpoint at step `30950`
- switch only the **data recipe** to the zzw-style 4-source mixture
- no teacher, pure hard CE

Data mixture in `v12`:

- FineWeb: `59%`
- Wikipedia: `17%`
- Science: `12%`
- Books: `12%`

Concrete paths:

- FineWeb: `tokenized_sources/fineweb_full/`
- Wikipedia: `/data/fengfei/cse251b-nanogpt-zzv-train/data/wikipedia/`
- Science: `/data/fengfei/cse251b-nanogpt-zzv-train/data/science/`
- Books: `/data/fengfei/cse251b-nanogpt-zzv-train/data/books/`

Why it worked:

- it keeps the strong student and training code from this repo
- but swaps in a more validation-like source mix
- it does not rely on a teacher at this stage

---

## 3. How to Run

### 3.1 Prepare the original two-source core mixture

This was the original path for FineWeb + Cosmopedia:

```bash
cd /data/fengfei/cse251b-nanogpt/build-nanogpt
python3 prepare_pretrain_data.py \
  --mode full \
  --dataset_mix_config data_configs/mix_full_no_target_16k.template.json \
  --target_num_tokens 0 \
  --max_documents_per_dataset 0 \
  --tokenizer_backend gpt2_tiktoken \
  --output_dir prepared_mixture_gpt2_full \
  --overwrite
```

### 3.2 Prepare extra standalone late-stage sources

OpenWebText:

```bash
cd /data/fengfei/cse251b-nanogpt/build-nanogpt
python3 prepare_v8_extra_source.py --preset openwebtext --target_tokens 2000000000 --overwrite
```

Wikipedia:

```bash
cd /data/fengfei/cse251b-nanogpt/build-nanogpt
python3 prepare_v8_extra_source.py --preset wikipedia --target_tokens 2000000000 --overwrite
```

### 3.3 Prepare zzw-style shard sources

These live in the external zzw repo:

```bash
cd /data/fengfei/cse251b-nanogpt-zzv-train/cse251b-nanogpt-zzv-train
python prepare_data.py --source wikipedia
python prepare_data.py --source science
python prepare_data.py --source books
```

### 3.4 Run the current best main-repo recipe (`v12`)

```bash
cd /data/fengfei/cse251b-nanogpt/build-nanogpt
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 python3 train_gpt2_v12.py
```

Current default `v12` schedule:

- pure CE
- resume from `v8@30950`
- `train_steps = 10000`
- `eval_interval = 1000`
- `save_interval = 1000`

---

## 4. Checkpoint Format

Most main-repo experiments export submission-ready directories of the form:

- `log/<version>/submission_YYYYMMDD_<exp_name>_<step>/`

Each such directory contains:

- `checkpoint.pt`
- `config.json`
- `model.py`
- `evaluate_results.json` (after official evaluation runs)

These are directly compatible with:

- `../evaluate.py`

---

## 5. Current Best Checkpoint

As of `2026-05-01`, the current best main-repo checkpoint is:

- `log/v12/submission_20260501_alvin_v12_40949`

Official validation result (`val.bin`):

- **Perplexity:** `22.2732`
- **Avg loss (nats):** `3.103385`
- **Tokens evaluated:** `5,169,152`

This is currently the best documented checkpoint in `build-nanogpt/log/`.

---

## 6. Practical Takeaways

1. The baseline (`v1`) established the main architecture and the first strong checkpoints.
2. `v4` showed that a better late-stage continuation schedule matters a lot.
3. Heavy hidden-state / aggressive KD experiments were usually not worth it late in training.
4. Light KD (`v8`) helped when paired with more validation-like data.
5. The biggest recent jump came from changing the **data recipe** rather than changing the model architecture.
6. `v12` is currently the best line because it combines:
   - a strong resumed checkpoint
   - stable pure CE continuation
   - a zzw-style validation-like data mix

