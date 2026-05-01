#!/usr/bin/env bash
set -euo pipefail

cd /data/fengfei/cse251b-nanogpt/build-nanogpt

source /data/fengfei/RLUQ/miniconda3/bin/activate /data/fengfei/RLUQ/envs/unsloth

# # Prepare full mixed training data (train.bin only).
# python3 prepare_pretrain_data.py \
#   --mode full \
#   --dataset_mix_config data_configs/mix_full_no_target_16k.template.json \
#   --target_num_tokens 0 \
#   --max_documents_per_dataset 0 \
#   --output_dir prepared_mixture_gpt2_full \
#   --overwrite

# Train on GPU 7. The training script will:
# - infer max_steps from train.bin size
# - save every 1000 steps
# - export a submission-ready folder each save
# - run official evaluate.py on each exported folder and log the result
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=6 python3 train_gpt2.py
