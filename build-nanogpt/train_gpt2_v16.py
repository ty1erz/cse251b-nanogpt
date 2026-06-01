#!/usr/bin/env python3
"""
train_gpt2_v16.py — v13 continuation training.

This is intentionally plain CE training: same model/data/loader style as v13,
but with a zzw-style low-LR resume schedule for a second epoch.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from muon import Muon, split_params_for_muon
from submission_model_template import GPT, GPTConfig


# ─── Paths ────────────────────────────────────────────────────────────────────
DEFAULT_PUBLIC_VAL = "/data/fengfei/cse251b-nanogpt/val.bin"
DEFAULT_EVAL_PY    = "/data/fengfei/cse251b-nanogpt/evaluate.py"
START_STEP = 0

V16_SOURCE_SPECS = [
    {"name": "fineweb",   "path": "/data/fengfei/cse251b-nanogpt/build-nanogpt/tokenized_sources/fineweb_full", "weight": 0.50},
    {"name": "wikipedia", "path": "/data/fengfei/cse251b-nanogpt-zzv-train/data/wikipedia", "weight": 0.20},
    {"name": "science",   "path": "/data/fengfei/cse251b-nanogpt-zzv-train/data/science",   "weight": 0.15},
    {"name": "books",     "path": "/data/fengfei/cse251b-nanogpt-zzv-train/data/books",     "weight": 0.15},
]

# ─── Training schedule ────────────────────────────────────────────────────────
MICRO_BATCH      = 16
SEQ_LEN          = 1024
TOTAL_BATCH_SIZE = 524288          # zzw-style 512 K tokens / step
TARGET_STEP      = 76000           # continue v13 38000 -> 76000, roughly 2 epochs
LR_DECAY_STEPS   = TARGET_STEP
WARMUP_STEPS     = 0
SAVE_INTERVAL    = 500             # full export + official eval + checkpoint save
EVAL_INTERVAL    = 500             # save step already runs official eval

# ─── LR ───────────────────────────────────────────────────────────────────────
MUON_LR       = 4e-4
MUON_MOMENTUM = 0.95
ADAM_LR       = 1.6e-5
MIN_LR_RATIO  = 0.5

WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0

EVAL_BATCH_SIZE = 8


# ─── Utility functions ────────────────────────────────────────────────────────

def count_params(model):
    seen, total = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def get_lr(schedule_step: int, peak: float, warmup_steps: int, decay_steps: int, min_lr_ratio: float) -> float:
    floor = peak * min_lr_ratio
    if warmup_steps > 0 and schedule_step < warmup_steps:
        return peak * (schedule_step + 1) / warmup_steps
    if schedule_step >= decay_steps:
        return floor
    ratio = (schedule_step - warmup_steps) / max(1, decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return floor + coeff * (peak - floor)


@torch.no_grad()
def run_in_memory_eval(model, data_path, device, amp_dtype, block_size=1024, batch_size=8):
    was_training = model.training
    model.eval()
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    data = torch.from_numpy(data.astype(np.int64))
    n_chunks = (len(data) - 1) // block_size
    n_chunks = (n_chunks // batch_size) * batch_size
    total_loss, total_tok = 0.0, 0
    dtype_str = "cuda" if str(device).startswith("cuda") else "cpu"
    for i in range(0, n_chunks, batch_size):
        x = torch.stack([data[j * block_size: j * block_size + block_size]
                         for j in range(i, i + batch_size)]).to(device)
        y = torch.stack([data[j * block_size + 1: j * block_size + block_size + 1]
                         for j in range(i, i + batch_size)]).to(device)
        with torch.autocast(device_type=dtype_str, dtype=amp_dtype):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
        total_loss += loss.item()
        total_tok += y.numel()
    if was_training:
        model.train()
    avg = total_loss / total_tok
    return {"perplexity": math.exp(avg), "avg_loss_nats": avg, "total_tokens_evaluated": total_tok}


def run_official_eval(model_dir, device, evaluate_py, public_val_bin):
    out_json = os.path.join(model_dir, "evaluate_results.json")
    cmd = [sys.executable, evaluate_py,
           "--model_dir", model_dir, "--data", public_val_bin,
           "--device", device, "--output_json", out_json]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    with open(out_json) as f:
        results = json.load(f)
    return results, time.time() - t0


# ─── Data loader ──────────────────────────────────────────────────────────────

class MultiSourceWindowLoader:
    def __init__(self, source_specs, B, T, process_rank, num_processes, seed=2025, pin_memory=False):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.pin_memory = pin_memory
        self.rng = np.random.default_rng(seed)
        self.sources = []
        self.source_counts = {}
        self.source_cursor = 0
        self.source_cycle = []

        missing = []
        for spec in source_specs:
            path = spec["path"]
            if not os.path.exists(path):
                missing.append((spec["name"], path))
                continue
            source = {
                "name": spec["name"], "path": path,
                "weight": float(spec["weight"]),
                "shard_i": 0, "pos": T * process_rank, "epochs": 0,
            }
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "train.bin")):
                tokens = np.memmap(os.path.join(path, "train.bin"), dtype=np.uint16, mode="r")
                start, end = int(spec.get("start", 0)), int(spec.get("end", len(tokens)))
                source.update({"mode": "train_bin", "shards": [tokens], "starts": [start], "ends": [end]})
            elif os.path.isdir(path):
                shard_paths = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npy"))
                if not shard_paths:
                    missing.append((spec["name"], path + " (no .npy shards or train.bin)"))
                    continue
                shard_tokens = [np.load(sp, mmap_mode="r") for sp in shard_paths]
                source.update({
                    "mode": "npy_shards", "shards": shard_tokens, "shard_paths": shard_paths,
                    "starts": [0] * len(shard_tokens), "ends": [len(a) for a in shard_tokens],
                })
            else:
                tokens = np.memmap(path, dtype=np.uint16, mode="r")
                start, end = int(spec.get("start", 0)), int(spec.get("end", len(tokens)))
                source.update({"mode": "train_bin", "shards": [tokens], "starts": [start], "ends": [end]})
            self.sources.append(source)
            self.source_counts[spec["name"]] = 0

        if missing:
            msg = "\n".join(f"  - {n}: {p}" for n, p in missing)
            raise FileNotFoundError(f"Missing tokenized sources:\n{msg}")

        weights = np.asarray([s["weight"] for s in self.sources], dtype=np.float64)
        self.weights = weights / weights.sum()
        self._build_source_cycle()

    def _build_source_cycle(self):
        cycle_len = 20
        counts = np.rint(self.weights * cycle_len).astype(int)
        while counts.sum() < cycle_len:
            counts[np.argmax(self.weights * cycle_len - counts)] += 1
        while counts.sum() > cycle_len:
            counts[np.argmax(counts - self.weights * cycle_len)] -= 1
        cycle = []
        for i, c in enumerate(counts):
            cycle.extend([i] * int(c))
        self.rng.shuffle(cycle)
        self.source_cycle = cycle
        self.source_cursor = 0

    def _next_source_idx(self):
        if self.source_cursor >= len(self.source_cycle):
            self._build_source_cycle()
        idx = self.source_cycle[self.source_cursor]
        self.source_cursor += 1
        return idx

    def _next_from_source(self, source):
        while True:
            shard = source["shards"][source["shard_i"]]
            start0 = source["starts"][source["shard_i"]]
            end = source["ends"][source["shard_i"]]
            if source["pos"] < start0 + self.T * self.process_rank:
                source["pos"] = start0 + self.T * self.process_rank
            if source["pos"] + self.T + 1 <= end:
                start = source["pos"]
                source["pos"] += self.T * self.num_processes
                return np.asarray(shard[start: start + self.T + 1], dtype=np.int64)
            source["shard_i"] += 1
            if source["shard_i"] >= len(source["shards"]):
                source["shard_i"] = 0
                source["epochs"] += 1
            source["pos"] = source["starts"][source["shard_i"]] + self.T * self.process_rank

    def consume_source_counts(self):
        counts = dict(self.source_counts)
        for k in self.source_counts:
            self.source_counts[k] = 0
        return counts

    def next_batch(self):
        x = torch.empty((self.B, self.T), dtype=torch.long, pin_memory=self.pin_memory)
        y = torch.empty((self.B, self.T), dtype=torch.long, pin_memory=self.pin_memory)
        for i in range(self.B):
            source = self.sources[int(self._next_source_idx())]
            buf = self._next_from_source(source)
            x[i] = torch.from_numpy(buf[:-1].copy())
            y[i] = torch.from_numpy(buf[1:].copy())
            self.source_counts[source["name"]] += 1
        return x, y

    def state_dict(self):
        return {
            "rng": self.rng.bit_generator.state,
            "process_rank": self.process_rank,
            "num_processes": self.num_processes,
            "source_cursor": self.source_cursor,
            "source_cycle": list(self.source_cycle),
            "sources": {s["name"]: {"shard_i": int(s["shard_i"]), "pos": int(s["pos"]), "epochs": int(s["epochs"])}
                        for s in self.sources},
            "source_counts": dict(self.source_counts),
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        saved_np = int(state.get("num_processes", self.num_processes))
        if saved_np != self.num_processes:
            raise ValueError(f"num_processes mismatch: checkpoint={saved_np}, current={self.num_processes}")
        self.source_cursor = int(state["source_cursor"])
        self.source_cycle = list(state["source_cycle"])
        rank_delta = (self.process_rank - int(state.get("process_rank", 0))) * self.T
        for source in self.sources:
            if source["name"] in state.get("sources", {}):
                saved = state["sources"][source["name"]]
                source["shard_i"] = int(saved["shard_i"])
                source["pos"] = int(saved["pos"]) + rank_delta
                source["epochs"] = int(saved.get("epochs", 0))
        self.source_counts.update(state.get("source_counts", {}))


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default=None, help="Override experiment name for logs/submissions")
    p.add_argument("--resume", default=None, help="Path to v13 train_ckpt_38000.pt or later")
    p.add_argument("--target_step", type=int, default=TARGET_STEP)
    p.add_argument("--micro_batch", type=int, default=MICRO_BATCH)
    p.add_argument("--total_batch_size", type=int, default=TOTAL_BATCH_SIZE)
    p.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE)
    p.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--lr_decay_steps", type=int, default=None)
    p.add_argument("--muon_lr", type=float, default=MUON_LR)
    p.add_argument("--adam_lr", type=float, default=ADAM_LR)
    p.add_argument("--min_lr_ratio", type=float, default=MIN_LR_RATIO)
    p.add_argument("--save_interval", type=int, default=SAVE_INTERVAL)
    p.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--global_lr_schedule", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── DDP setup ──────────────────────────────────────────────────────────────
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend="nccl")
        ddp_rank       = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device         = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2025)
    torch.set_float32_matmul_precision("high")

    B, T = args.micro_batch, SEQ_LEN
    total_batch_size = args.total_batch_size
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    target_step = args.target_step
    lr_decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else target_step
    if master_process:
        print(f"total_batch_size={total_batch_size}, grad_accum_steps={grad_accum_steps}")

    # ── Data loader ────────────────────────────────────────────────────────────
    train_loader = MultiSourceWindowLoader(
        source_specs=V16_SOURCE_SPECS, B=B, T=T,
        process_rank=ddp_rank, num_processes=ddp_world_size,
        seed=2025, pin_memory=(device_type == "cuda"),
    )

    if args.resume is None:
        raise ValueError("v16 is v13 continuation-only. Pass --resume log/v13/train_ckpt_38000.pt")

    # ── Load v13 checkpoint ────────────────────────────────────────────────────
    start_step = START_STEP
    initial_step = START_STEP
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model_config = GPTConfig(**ckpt["config"])
    model = GPT(model_config)
    model.load_state_dict(ckpt["model"], strict=True)
    train_loader.load_state_dict(ckpt["train_loader"])
    start_step = int(ckpt["step"])
    initial_step = int(ckpt.get("initial_step", start_step))

    if target_step <= start_step:
        raise ValueError(f"target_step ({target_step}) must be > start_step ({start_step})")
    if start_step < initial_step:
        raise ValueError(f"start_step ({start_step}) must be >= initial_step ({initial_step})")
    model.to(device)
    model_config.exp_name = args.run_name or "alvin_v16_v13_continue"
    raw_model = model

    amp_dtype = (
        torch.bfloat16 if device_type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device_type == "cuda"
        else torch.float32
    )

    # ── DDP wrap ───────────────────────────────────────────────────────────────
    if ddp:
        train_model = DDP(raw_model, device_ids=[ddp_local_rank])
    else:
        train_model = raw_model

    # ── Optimiser ──────────────────────────────────────────────────────────────
    muon_params, adam_decay, adam_nodecay = split_params_for_muon(raw_model)
    muon_optim = Muon(
        muon_params,
        lr=args.muon_lr, momentum=MUON_MOMENTUM, nesterov=True,
        ns_steps=5, weight_decay=WEIGHT_DECAY,
    )
    adam_optim = torch.optim.AdamW(
        [
            {"params": adam_decay,   "weight_decay": 0.0, "lr": args.adam_lr},
            {"params": adam_nodecay, "weight_decay": 0.0, "lr": args.adam_lr},
        ],
        lr=args.adam_lr, betas=(0.9, 0.95), eps=1e-8,
        fused=(device_type == "cuda"),
    )

    if args.resume:
        muon_optim.load_state_dict(ckpt["muon_optim"])
        adam_optim.load_state_dict(ckpt["adam_optim"])
        for grp in adam_optim.param_groups:
            grp["weight_decay"] = 0.0
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        if "cuda_rng" in ckpt and ckpt["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())

    if master_process:
        print(f"resumed v13 checkpoint from {args.resume} at step {start_step}")
        print(f"student params: {count_params(raw_model):,}")
        print(f"Muon  → {len(muon_params)} tensors, {sum(p.numel() for p in muon_params):,} params")
        print(f"AdamW → {len(adam_decay)+len(adam_nodecay)} tensors, weight_decay=0")
        print(
            f"LR: muon {args.muon_lr:.1e}→{args.muon_lr * args.min_lr_ratio:.1e}  "
            f"adam {args.adam_lr:.1e}→{args.adam_lr * args.min_lr_ratio:.1e}  "
            f"warmup={args.warmup_steps}  decay={lr_decay_steps}  "
            f"global_schedule={args.global_lr_schedule}"
        )
        for s in train_loader.sources:
            print(f"  {s['name']}: weight={s['weight']:.2f}")

    # ── Logging / checkpoint helpers ───────────────────────────────────────────
    submission_config_fields = (
        "block_size", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd",
        "dropout", "bias", "norm_type", "position_embedding", "mlp_type",
        "intermediate_size", "tie_embeddings",
    )
    log_dir  = os.path.join("log", "v16")
    os.makedirs(log_dir, exist_ok=True)
    run_date = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"log_{run_date}_{model_config.exp_name}.txt")
    submission_template = os.path.join(os.path.dirname(__file__), "submission_model_template.py")

    with open(log_file, "a" if args.resume else "w") as f:
        f.write(
            f"# v13 continuation from {args.resume} at step {start_step}, "
            f"initial_step {initial_step}, target {target_step}, "
            f"total_batch_size {total_batch_size}\n"
        )

    def export_submission_bundle(model_obj, step):
        """Export raw v13-continuation model weights."""
        out_dir = os.path.join(log_dir, f"submission_{run_date}_{model_config.exp_name}_{step:05d}")
        os.makedirs(out_dir, exist_ok=True)
        cfg = {k: getattr(model_obj.config, k) for k in submission_config_fields}
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        shutil.copyfile(submission_template, os.path.join(out_dir, "model.py"))
        torch.save(model_obj.state_dict(), os.path.join(out_dir, "checkpoint.pt"))
        return out_dir

    def save_train_checkpoint(step):
        ckpt_path = os.path.join(log_dir, f"train_ckpt_{step:05d}.pt")
        payload = {
            "model":        raw_model.state_dict(),
            "muon_optim":   muon_optim.state_dict(),
            "adam_optim":   adam_optim.state_dict(),
            "train_loader": train_loader.state_dict(),
            "torch_rng":    torch.get_rng_state(),
            "cuda_rng":     torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "config":       {k: getattr(raw_model.config, k) for k in submission_config_fields},
            "step":         step,
            "initial_step": initial_step,
        }
        torch.save(payload, ckpt_path)
        shutil.copyfile(ckpt_path, os.path.join(log_dir, "train_ckpt_latest.pt"))
        return ckpt_path

    # ── Training loop ──────────────────────────────────────────────────────────
    vocab_size = raw_model.config.vocab_size

    for step in range(start_step + 1, target_step + 1):
        schedule_step = step if args.global_lr_schedule else step - initial_step - 1
        last_step = step == target_step
        t0        = time.time()

        train_model.train()
        muon_optim.zero_grad(set_to_none=True)
        adam_optim.zero_grad(set_to_none=True)

        # ── CE accumulation ───────────────────────────────────────────────────
        ce_loss_accum = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if ddp:
                train_model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                logits = train_model(x)
                ce = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            (ce / grad_accum_steps).backward()
            ce_loss_accum += ce.detach() / grad_accum_steps

        # ── Optimiser step ────────────────────────────────────────────────────
        if ddp:
            dist.all_reduce(ce_loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)

        muon_lr = get_lr(
            schedule_step, args.muon_lr, args.warmup_steps,
            lr_decay_steps, args.min_lr_ratio,
        )
        adam_lr = get_lr(
            schedule_step, args.adam_lr, args.warmup_steps,
            lr_decay_steps, args.min_lr_ratio,
        )
        for grp in muon_optim.param_groups:
            grp["lr"] = muon_lr
        for grp in adam_optim.param_groups:
            grp["lr"] = adam_lr

        muon_optim.step()
        adam_optim.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_per_sec = B * T * grad_accum_steps * ddp_world_size / dt

        save_due = step > start_step and ((step - start_step) % args.save_interval == 0 or last_step)
        eval_due = step > start_step and (step - start_step) % args.eval_interval == 0 and not save_due

        if master_process:
            print(
                f"step {step:5d} | ce {ce_loss_accum.item():.6f} | "
                f"muon {muon_lr:.2e} adam {adam_lr:.2e} | "
                f"norm {norm:.4f} | dt {dt*1000:.0f}ms | {tokens_per_sec:.0f} tok/s"
            )
            with open(log_file, "a") as f:
                f.write(
                    f"{step} train ce {ce_loss_accum.item():.6f} "
                    f"muon_lr {muon_lr:.8e} adam_lr {adam_lr:.8e}\n"
                )

            if eval_due:
                quick = run_in_memory_eval(
                    raw_model, DEFAULT_PUBLIC_VAL, device, amp_dtype,
                    batch_size=args.eval_batch_size,
                )
                print(f"quick_eval | ppl {quick['perplexity']:.4f} | loss {quick['avg_loss_nats']:.6f}")
                with open(log_file, "a") as f:
                    f.write(
                        f"{step} quick_eval ppl {quick['perplexity']:.4f} "
                        f"loss {quick['avg_loss_nats']:.6f} "
                        f"tok {quick['total_tokens_evaluated']}\n"
                    )

            if save_due:
                counts = train_loader.consume_source_counts()
                msg = " ".join(f"{k}={v}" for k, v in counts.items())
                ckpt_path = save_train_checkpoint(step)
                export_dir = export_submission_bundle(raw_model, step)
                eval_error = None
                try:
                    results, elapsed = run_official_eval(
                        export_dir, device, DEFAULT_EVAL_PY, DEFAULT_PUBLIC_VAL
                    )
                except subprocess.CalledProcessError as exc:
                    results, elapsed = None, 0.0
                    eval_error = f"official_eval_failed returncode={exc.returncode}"
                with open(log_file, "a") as f:
                    f.write(f"{step} sampled_windows {msg}\n")
                    f.write(f"{step} train_ckpt {ckpt_path}\n")
                    f.write(f"{step} export {export_dir}\n")
                    if results is None:
                        f.write(f"{step} evaluate_failed {eval_error}\n")
                    else:
                        f.write(
                            f"{step} evaluate ppl {results['perplexity']:.4f} "
                            f"loss {results['avg_loss_nats']:.6f} "
                            f"tok {results['total_tokens_evaluated']} sec {elapsed:.1f}\n"
                        )
                if results is None:
                    print(f"eval failed after saving {export_dir}: {eval_error}")
                else:
                    print(
                        f"eval | ppl {results['perplexity']:.4f} | "
                        f"loss {results['avg_loss_nats']:.6f} | "
                        f"tok {results['total_tokens_evaluated']:,} | {elapsed:.0f}s"
                    )

        if ddp and (save_due or eval_due):
            dist.barrier()

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
