#!/usr/bin/env python3
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from submission_model_template import GPT, GPTConfig

FINEWEB_TOTAL_TOKENS = 9953989344
COSMOPEDIA_TOTAL_TOKENS = 3624349131
TOTAL_BATCH_SIZE = 262144
DEFAULT_SUBMISSION_DIR = "/data/fengfei/cse251b-nanogpt/build-nanogpt/log/v5_distill/submission_20260427_alvin_v5_distill_26250"
DEFAULT_TRAIN_BIN = "/data/fengfei/cse251b-nanogpt/build-nanogpt/prepared_mixture_gpt2_full/train.bin"
DEFAULT_OUT_DIR = "/data/fengfei/cse251b-nanogpt/build-nanogpt/v7_pools"

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def progress_iter(iterable, total=None, desc="progress"):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


def count_params(model):
    seen = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += p.numel()
    return total


def load_student(submission_dir: str, device: str):
    config_path = os.path.join(submission_dir, "config.json")
    ckpt_path = os.path.join(submission_dir, "checkpoint.pt")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = GPTConfig(**json.load(f))
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model = GPT(cfg)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def sample_unique_starts(rng, low: int, high_inclusive: int, n: int):
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    starts = set()
    if tqdm is None:
        print(f"[sampling] target={n:,} range=[{low:,}, {high_inclusive:,}]")
    while len(starts) < n:
        needed = n - len(starts)
        draw = rng.integers(low, high_inclusive + 1, size=needed * 2, dtype=np.int64)
        starts.update(int(x) for x in draw)
    arr = np.fromiter(list(starts), dtype=np.int64)
    rng.shuffle(arr)
    return arr[:n]


def build_batch(tokens, starts, block_size, device):
    bsz = len(starts)
    x = torch.empty((bsz, block_size), dtype=torch.long)
    y = torch.empty((bsz, block_size), dtype=torch.long)
    for i, start in enumerate(starts):
        buf = np.asarray(tokens[start : start + block_size + 1], dtype=np.int64)
        x[i] = torch.from_numpy(buf[:-1].copy())
        y[i] = torch.from_numpy(buf[1:].copy())
    return x.to(device), y.to(device)


@torch.no_grad()
def score_candidates(student, teacher, tokens, starts, source_name, block_size, batch_size, device, amp_dtype):
    teacher_nlls = []
    student_nlls = []
    total = len(starts)
    batch_starts_iter = range(0, total, batch_size)
    total_batches = (total + batch_size - 1) // batch_size
    for i in progress_iter(batch_starts_iter, total=total_batches, desc=f"scoring:{source_name}"):
        batch_starts = starts[i : i + batch_size]
        x, y = build_batch(tokens, batch_starts, block_size, device)
        with torch.autocast(device_type=("cuda" if device.startswith("cuda") else "cpu"), dtype=amp_dtype):
            student_logits = student(x)
            teacher_logits = teacher(x).logits
        vocab_s = student_logits.size(-1)
        vocab_t = teacher_logits.size(-1)
        student_loss = F.cross_entropy(
            student_logits.reshape(-1, vocab_s),
            y.reshape(-1),
            reduction="none",
        ).view(x.size(0), block_size).mean(dim=1)
        teacher_loss = F.cross_entropy(
            teacher_logits.reshape(-1, vocab_t),
            y.reshape(-1),
            reduction="none",
        ).view(x.size(0), block_size).mean(dim=1)
        student_nlls.append(student_loss.detach().float().cpu().numpy())
        teacher_nlls.append(teacher_loss.detach().float().cpu().numpy())
    teacher_nll = np.concatenate(teacher_nlls, axis=0)
    student_nll = np.concatenate(student_nlls, axis=0)
    gap = student_nll - teacher_nll
    source_ids = np.zeros_like(starts, dtype=np.int64) if source_name == "fineweb_remaining" else np.ones_like(starts, dtype=np.int64)
    return {
        "source_name": source_name,
        "source_id": source_ids,
        "start_offset": starts.astype(np.int64),
        "teacher_nll": teacher_nll.astype(np.float32),
        "student_nll": student_nll.astype(np.float32),
        "gap": gap.astype(np.float32),
    }


def select_indices(metric, keep_top_fraction, largest: bool):
    k = max(1, int(len(metric) * keep_top_fraction))
    if largest:
        order = np.argsort(-metric)
    else:
        order = np.argsort(metric)
    return order[:k]


def main():
    parser = argparse.ArgumentParser(description="Build V7 GPT-2 XL guided training window pool")
    parser.add_argument("--submission_dir", type=str, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--train_bin", type=str, default=DEFAULT_TRAIN_BIN)
    parser.add_argument("--resume_step", type=int, default=26250)
    parser.add_argument("--teacher_model", type=str, default="gpt2-xl")
    parser.add_argument("--candidate_windows_per_source", type=int, default=5000)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--score_batch_size", type=int, default=8)
    parser.add_argument("--selection_mode", type=str, default="student_teacher_gap", choices=["student_teacher_gap", "teacher_low_nll"])
    parser.add_argument("--keep_top_fraction", type=float, default=0.5)
    parser.add_argument("--source_balanced", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("V7 data selection")
    print(f"submission_dir={args.submission_dir}")
    print(f"teacher_model={args.teacher_model}")
    print(f"candidate_windows_per_source={args.candidate_windows_per_source:,}")
    print(f"selection_mode={args.selection_mode}")
    print(f"keep_top_fraction={args.keep_top_fraction}")
    print(f"source_balanced={args.source_balanced}")
    print("=" * 60)

    used_tokens = args.resume_step * TOTAL_BATCH_SIZE
    fineweb_low = used_tokens
    fineweb_high = FINEWEB_TOTAL_TOKENS - (args.block_size + 1)
    cosmo_low = FINEWEB_TOTAL_TOKENS
    cosmo_high = FINEWEB_TOTAL_TOKENS + COSMOPEDIA_TOTAL_TOKENS - (args.block_size + 1)

    rng = np.random.default_rng(args.seed)
    print("[1/5] sampling candidate start offsets...")
    fineweb_starts = sample_unique_starts(rng, fineweb_low, fineweb_high, args.candidate_windows_per_source)
    cosmo_starts = sample_unique_starts(rng, cosmo_low, cosmo_high, args.candidate_windows_per_source)

    tokens = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    print("[2/5] loading student checkpoint...")
    student, cfg = load_student(args.submission_dir, args.device)
    amp_dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if args.device == "cuda" else torch.float32)
    print("[3/5] loading GPT-2 XL teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=amp_dtype if args.device == "cuda" else torch.float32)
    teacher.to(args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student params: {count_params(student):,}")
    print(f"Teacher params: {count_params(teacher):,}")
    print(f"Scoring {args.candidate_windows_per_source:,} windows/source | mode={args.selection_mode} | keep_top_fraction={args.keep_top_fraction}")

    t0 = time.time()
    print("[4/5] scoring FineWeb candidates...")
    fineweb = score_candidates(student, teacher, tokens, fineweb_starts, "fineweb_remaining", args.block_size, args.score_batch_size, args.device, amp_dtype)
    print("[4/5] scoring Cosmopedia candidates...")
    cosmopedia = score_candidates(student, teacher, tokens, cosmo_starts, "cosmopedia", args.block_size, args.score_batch_size, args.device, amp_dtype)
    score_elapsed = time.time() - t0

    all_source_id = np.concatenate([fineweb["source_id"], cosmopedia["source_id"]])
    all_start = np.concatenate([fineweb["start_offset"], cosmopedia["start_offset"]])
    all_teacher = np.concatenate([fineweb["teacher_nll"], cosmopedia["teacher_nll"]])
    all_student = np.concatenate([fineweb["student_nll"], cosmopedia["student_nll"]])
    all_gap = np.concatenate([fineweb["gap"], cosmopedia["gap"]])

    if args.selection_mode == "student_teacher_gap":
        metric = all_gap
        largest = True
    else:
        metric = all_teacher
        largest = False

    if args.source_balanced:
        selected_parts = []
        for source_id in [0, 1]:
            mask = all_source_id == source_id
            idx_local = select_indices(metric[mask], args.keep_top_fraction, largest=largest)
            global_idx = np.flatnonzero(mask)[idx_local]
            selected_parts.append(global_idx)
        selected_idx = np.concatenate(selected_parts)
    else:
        selected_idx = select_indices(metric, args.keep_top_fraction, largest=largest)

    print("[5/5] saving candidate scores and selected pool...")
    selected = {
        "source_id": all_source_id[selected_idx].astype(np.int64),
        "start_offset": all_start[selected_idx].astype(np.int64),
        "teacher_nll": all_teacher[selected_idx].astype(np.float32),
        "student_nll": all_student[selected_idx].astype(np.float32),
        "gap": all_gap[selected_idx].astype(np.float32),
    }

    candidates_path = os.path.join(args.output_dir, "candidates_v7_latest.npz")
    selected_path = os.path.join(args.output_dir, "selected_pool_v7_latest.npz")
    np.savez_compressed(
        candidates_path,
        source_id=all_source_id,
        start_offset=all_start,
        teacher_nll=all_teacher,
        student_nll=all_student,
        gap=all_gap,
    )
    np.savez_compressed(selected_path, **selected)

    summary = {
        "submission_dir": args.submission_dir,
        "resume_step": args.resume_step,
        "teacher_model": args.teacher_model,
        "candidate_windows_per_source": args.candidate_windows_per_source,
        "selection_mode": args.selection_mode,
        "keep_top_fraction": args.keep_top_fraction,
        "source_balanced": args.source_balanced,
        "seed": args.seed,
        "score_batch_size": args.score_batch_size,
        "block_size": args.block_size,
        "score_elapsed_sec": round(score_elapsed, 2),
        "student_params": count_params(student),
        "teacher_params": count_params(teacher),
        "per_source": {
            "fineweb_remaining": {
                "candidate_count": int(len(fineweb_starts)),
                "teacher_nll_mean": float(fineweb["teacher_nll"].mean()),
                "student_nll_mean": float(fineweb["student_nll"].mean()),
                "gap_mean": float(fineweb["gap"].mean()),
                "selected_count": int((selected["source_id"] == 0).sum()),
            },
            "cosmopedia": {
                "candidate_count": int(len(cosmo_starts)),
                "teacher_nll_mean": float(cosmopedia["teacher_nll"].mean()),
                "student_nll_mean": float(cosmopedia["student_nll"].mean()),
                "gap_mean": float(cosmopedia["gap"].mean()),
                "selected_count": int((selected["source_id"] == 1).sum()),
            },
        },
        "paths": {
            "candidates_npz": candidates_path,
            "selected_pool_npz": selected_path,
        },
    }
    summary_path = os.path.join(args.output_dir, "summary_v7_latest.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Saved candidates: {candidates_path}")
    print(f"Saved selected pool: {selected_path}")
    print(f"Saved summary: {summary_path}")
    for src, stats in summary["per_source"].items():
        print(
            f"{src}: cand={stats['candidate_count']:,} sel={stats['selected_count']:,} "
            f"teacher_nll={stats['teacher_nll_mean']:.4f} student_nll={stats['student_nll_mean']:.4f} gap={stats['gap_mean']:.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
