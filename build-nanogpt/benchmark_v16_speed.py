#!/usr/bin/env python3
"""
Benchmark v16 training-step speed on one GPU.

This runs the same core work as train_gpt2_v16.py:
  - v13/v16 model loaded from a train checkpoint
  - v16 data mix and loader state
  - CE loss only
  - Muon + AdamW optimizer step
  - total_batch_size=524288, micro_batch=16, seq_len=1024 by default

It does not save checkpoints or run validation. Use CUDA_VISIBLE_DEVICES to pick
the physical GPU, then compare avg step time / tok/s across A100 and L40.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F

import train_gpt2_v16 as v16
from muon import Muon, split_params_for_muon
from submission_model_template import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", default="log/v13/train_ckpt_38000.pt")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--measure_steps", type=int, default=10)
    p.add_argument("--micro_batch", type=int, default=v16.MICRO_BATCH)
    p.add_argument("--total_batch_size", type=int, default=v16.TOTAL_BATCH_SIZE)
    p.add_argument("--target_step", type=int, default=v16.TARGET_STEP)
    p.add_argument("--lr_decay_steps", type=int, default=None)
    p.add_argument("--muon_lr", type=float, default=v16.MUON_LR)
    p.add_argument("--adam_lr", type=float, default=v16.ADAM_LR)
    p.add_argument("--min_lr_ratio", type=float, default=v16.MIN_LR_RATIO)
    p.add_argument("--sched_warmup_steps", type=int, default=v16.WARMUP_STEPS)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for A100/L40 benchmarking.")

    device = "cuda"
    device_type = "cuda"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_mem_gb = props.total_memory / 1024**3

    B = args.micro_batch
    T = v16.SEQ_LEN
    grad_accum_steps = args.total_batch_size // (B * T)
    if grad_accum_steps < 1 or grad_accum_steps * B * T != args.total_batch_size:
        raise ValueError(
            f"total_batch_size={args.total_batch_size} must be divisible by "
            f"micro_batch*seq_len={B*T}"
        )

    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model_config = GPTConfig(**ckpt["config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.train()

    train_loader = v16.MultiSourceWindowLoader(
        source_specs=v16.V16_SOURCE_SPECS,
        B=B,
        T=T,
        process_rank=0,
        num_processes=1,
        seed=args.seed,
        pin_memory=True,
    )
    train_loader.load_state_dict(ckpt["train_loader"])

    muon_params, adam_decay, adam_nodecay = split_params_for_muon(model)
    muon_optim = Muon(
        muon_params,
        lr=args.muon_lr,
        momentum=v16.MUON_MOMENTUM,
        nesterov=True,
        ns_steps=5,
        weight_decay=v16.WEIGHT_DECAY,
    )
    adam_optim = torch.optim.AdamW(
        [
            {"params": adam_decay, "weight_decay": 0.0, "lr": args.adam_lr},
            {"params": adam_nodecay, "weight_decay": 0.0, "lr": args.adam_lr},
        ],
        lr=args.adam_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    if "muon_optim" in ckpt:
        muon_optim.load_state_dict(ckpt["muon_optim"])
    if "adam_optim" in ckpt:
        adam_optim.load_state_dict(ckpt["adam_optim"])
        for group in adam_optim.param_groups:
            group["weight_decay"] = 0.0

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    vocab_size = model.config.vocab_size
    lr_decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else args.target_step
    start_step = int(ckpt.get("step", 0))
    tokens_per_step = B * T * grad_accum_steps

    print("=" * 70)
    print(f"GPU: {gpu_name} ({total_mem_gb:.1f} GiB)")
    print(f"torch: {torch.__version__}")
    print(f"resume: {args.resume}")
    print(f"ckpt step: {start_step}")
    print(f"micro_batch={B} seq_len={T} grad_accum_steps={grad_accum_steps}")
    print(f"tokens/step={tokens_per_step:,}")
    print(f"amp_dtype={amp_dtype}")
    print(f"warmup_steps={args.warmup_steps} measure_steps={args.measure_steps}")
    print("=" * 70)

    step_times = []
    losses = []
    total_iters = args.warmup_steps + args.measure_steps

    for bench_i in range(total_iters):
        global_step = start_step + bench_i + 1
        schedule_step = global_step
        muon_lr = v16.get_lr(
            schedule_step,
            args.muon_lr,
            args.sched_warmup_steps,
            lr_decay_steps,
            args.min_lr_ratio,
        )
        adam_lr = v16.get_lr(
            schedule_step,
            args.adam_lr,
            args.sched_warmup_steps,
            lr_decay_steps,
            args.min_lr_ratio,
        )
        for group in muon_optim.param_groups:
            group["lr"] = muon_lr
        for group in adam_optim.param_groups:
            group["lr"] = adam_lr

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        muon_optim.zero_grad(set_to_none=True)
        adam_optim.zero_grad(set_to_none=True)
        loss_accum = torch.tensor(0.0, device=device)

        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            (loss / grad_accum_steps).backward()
            loss_accum += loss.detach() / grad_accum_steps

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), v16.GRAD_CLIP)
        muon_optim.step()
        adam_optim.step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_s = tokens_per_step / dt
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

        is_warmup = bench_i < args.warmup_steps
        tag = "warmup" if is_warmup else "measure"
        print(
            f"{tag:7s} {bench_i + 1:03d}/{total_iters:03d} "
            f"dt={dt*1000:.1f}ms tok/s={tok_s:.0f} "
            f"loss={loss_accum.item():.6f} norm={float(norm):.4f} "
            f"muon={muon_lr:.8e} adam={adam_lr:.8e} peak_mem={peak_mem_gb:.2f}GiB"
        )

        if not is_warmup:
            step_times.append(dt)
            losses.append(float(loss_accum.item()))

    avg_dt = statistics.mean(step_times)
    std_dt = statistics.pstdev(step_times) if len(step_times) > 1 else 0.0
    avg_tok_s = tokens_per_step / avg_dt
    result = {
        "gpu": gpu_name,
        "torch": torch.__version__,
        "resume": args.resume,
        "ckpt_step": start_step,
        "micro_batch": B,
        "seq_len": T,
        "grad_accum_steps": grad_accum_steps,
        "tokens_per_step": tokens_per_step,
        "measure_steps": args.measure_steps,
        "avg_step_ms": avg_dt * 1000,
        "std_step_ms": std_dt * 1000,
        "avg_tok_per_sec": avg_tok_s,
        "avg_loss": statistics.mean(losses),
        "peak_memory_gib_last_step": torch.cuda.max_memory_allocated() / 1024**3,
    }

    print("=" * 70)
    print(
        f"RESULT gpu={gpu_name} avg_step={result['avg_step_ms']:.1f}ms "
        f"std={result['std_step_ms']:.1f}ms tok/s={avg_tok_s:.0f} "
        f"avg_loss={result['avg_loss']:.6f}"
    )
    print("=" * 70)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
