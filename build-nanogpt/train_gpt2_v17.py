#!/usr/bin/env python3
"""
train_gpt2_v17.py — staged v13 fine-tuning.

This keeps the v13/v16 model, data mix, batch size, loader, Muon/AdamW split,
and checkpoint format. The post-38k schedule borrows zzw's LR scale, then keeps
training longer with a small number of sequential LR drops.
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

V17_SOURCE_SPECS = [
    {"name": "fineweb",   "path": "/data/fengfei/cse251b-nanogpt/build-nanogpt/tokenized_sources/fineweb_full", "weight": 0.50},
    {"name": "wikipedia", "path": "/data/fengfei/cse251b-nanogpt-zzv-train/data/wikipedia", "weight": 0.20},
    {"name": "science",   "path": "/data/fengfei/cse251b-nanogpt-zzv-train/data/science",   "weight": 0.15},
    {"name": "books",     "path": "/data/fengfei/cse251b-nanogpt-zzv-train/data/books",     "weight": 0.15},
]

# ─── Training schedule ────────────────────────────────────────────────────────
MICRO_BATCH      = 16
SEQ_LEN          = 1024
TOTAL_BATCH_SIZE = 524288          # zzw-style 512 K tokens / step
TARGET_STEP      = 42000           # single-stage fallback target
LR_DECAY_STEPS   = TARGET_STEP
WARMUP_STEPS     = 200
SAVE_INTERVAL    = 500             # full export + official eval + checkpoint save
EVAL_INTERVAL    = 250             # save step already runs official eval

# ─── LR ───────────────────────────────────────────────────────────────────────
MUON_LR       = 4e-4
MUON_MOMENTUM = 0.95
ADAM_LR       = 1.6e-5
MIN_LR_RATIO  = 0.5

WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0

EVAL_BATCH_SIZE = 8


V17_FRIEND_STAGES = [
    # These stages mirror the CLI values from zzw's train_final_v1.py.
    # This script saves checkpoints after completing a step, so e.g. internal
    # train_ckpt_42000 is the practical counterpart of zzw's model_041999.pt.
    {
        "name": "ft_42k",
        "resume_stage": None,
        "resume_step": None,
        "target_step": 42000,
        "muon_lr": 6e-3,
        "adam_lr": 2.4e-4,
        "min_lr_ratio": 0.1,
        "save_interval": 500,
        "eval_interval": 250,
    },
    {
        "name": "ft_44k",
        "resume_stage": "ft_42k",
        "resume_step": 42000,
        "target_step": 44000,
        "muon_lr": 4e-3,
        "adam_lr": 1.8e-4,
        "min_lr_ratio": 0.1,
        "save_interval": 500,
        "eval_interval": 250,
    },
    {
        "name": "ft_45k_from43500",
        "resume_stage": "ft_44k",
        "resume_step": 43500,
        "target_step": 45000,
        "muon_lr": 3e-3,
        "adam_lr": 1.35e-4,
        "min_lr_ratio": 0.1,
        "save_interval": 250,
        "eval_interval": 250,
    },
    {
        "name": "ft_45750_from44750",
        "resume_stage": "ft_45k_from43500",
        "resume_step": 44750,
        "target_step": 45750,
        "muon_lr": 1.5e-3,
        "adam_lr": 8e-5,
        "min_lr_ratio": 0.1,
        "save_interval": 250,
        "eval_interval": 250,
    },
    {
        "name": "ft_46750_from45749",
        "resume_stage": "ft_45750_from44750",
        "resume_step": 45750,
        "target_step": 46750,
        "muon_lr": 1.2e-3,
        "adam_lr": 6.4e-5,
        "min_lr_ratio": 0.1,
        "save_interval": 250,
        "eval_interval": 250,
    },
    {
        "name": "ft_47750_from46749",
        "resume_stage": "ft_46750_from45749",
        "resume_step": 46750,
        "target_step": 47750,
        "muon_lr": 9e-4,
        "adam_lr": 5e-5,
        "min_lr_ratio": 0.1,
        "save_interval": 250,
        "eval_interval": 250,
    },
    {
        "name": "ft_51500_from47749",
        "resume_stage": "ft_47750_from46749",
        "resume_step": 47750,
        "target_step": 51500,
        "muon_lr": 7e-4,
        "adam_lr": 4e-5,
        "min_lr_ratio": 0.1,
        "save_interval": 750,
        "eval_interval": 750,
    },
]


V17_ADAPTIVE_STAGES = [
    # Fewer stages than zzw's exact CLI. The peak LR values stay in the same
    # family, and the default global cosine schedule makes the effective resume
    # LR close to zzw's actual fine-tuning LR near 38k+. Stages are sequential:
    # every stage resumes from the previous stage's final checkpoint.
    {
        "name": "boost_42k",
        "resume_stage": None,
        "resume_step": None,
        "target_step": 42000,
        "muon_lr": 6e-3,
        "adam_lr": 2.4e-4,
        "min_lr_ratio": 0.1,
        "save_interval": 500,
        "eval_interval": 250,
    },
    {
        "name": "settle_50k",
        "resume_stage": "boost_42k",
        "resume_step": 42000,
        "target_step": 50000,
        "muon_lr": 3e-3,
        "adam_lr": 1.35e-4,
        "min_lr_ratio": 0.1,
        "save_interval": 1000,
        "eval_interval": 500,
    },
    {
        "name": "polish_62k",
        "resume_stage": "settle_50k",
        "resume_step": 50000,
        "target_step": 62000,
        "muon_lr": 1.2e-3,
        "adam_lr": 6.4e-5,
        "min_lr_ratio": 0.1,
        "save_interval": 2000,
        "eval_interval": 1000,
    },
    {
        "name": "finish_76k",
        "resume_stage": "polish_62k",
        "resume_step": 62000,
        "target_step": 76000,
        "muon_lr": 7e-4,
        "adam_lr": 4e-5,
        "min_lr_ratio": 0.1,
        "save_interval": 2000,
        "eval_interval": 1000,
    },
]


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
    p.add_argument(
        "--stage_preset",
        choices=("adaptive", "friend", "single"),
        default="adaptive",
        help=(
            "adaptive uses four sequential LR-drop stages inspired by zzw; "
            "friend reproduces the full CLI stage list; single behaves like v16."
        ),
    )
    p.add_argument(
        "--branch_mode",
        choices=("best", "cli", "sequential"),
        default="sequential",
        help=(
            "best chooses the previous stage's lowest official-eval PPL checkpoint; "
            "cli uses the hard-coded zzw resume step; sequential uses the previous stage's final checkpoint."
        ),
    )
    p.add_argument("--start_stage", default=None, help="Optional stage name to start from inside the preset")
    p.add_argument("--stop_stage", default=None, help="Optional stage name to stop after inside the preset")
    p.add_argument("--log_root", default=os.path.join("log", "v17"))
    p.add_argument("--val_bin_path", default=DEFAULT_PUBLIC_VAL)
    p.add_argument("--evaluate_py", default=DEFAULT_EVAL_PY)
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
    if total_batch_size % (B * T * ddp_world_size) != 0:
        raise ValueError(
            f"total_batch_size={total_batch_size} must divide "
            f"micro_batch*seq_len*world_size={B*T*ddp_world_size}"
        )
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total_batch_size={total_batch_size}, grad_accum_steps={grad_accum_steps}")

    # ── Data loader ────────────────────────────────────────────────────────────
    train_loader = MultiSourceWindowLoader(
        source_specs=V17_SOURCE_SPECS, B=B, T=T,
        process_rank=ddp_rank, num_processes=ddp_world_size,
        seed=2025, pin_memory=(device_type == "cuda"),
    )

    if args.resume is None:
        raise ValueError("v17 is continuation-only. Pass --resume log/v13/train_ckpt_38000.pt")

    # ── Build model from the initial checkpoint config ─────────────────────────
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model_config = GPTConfig(**ckpt["config"])
    model = GPT(model_config)
    model.to(device)
    model_config.exp_name = args.run_name or "alvin_v17_multistage"
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

    loaded_checkpoint_stage = None

    def load_training_checkpoint(path, preloaded=None):
        nonlocal loaded_checkpoint_stage
        state = preloaded if preloaded is not None else torch.load(path, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"], strict=True)
        train_loader.load_state_dict(state["train_loader"])
        muon_optim.load_state_dict(state["muon_optim"])
        adam_optim.load_state_dict(state["adam_optim"])
        for grp in adam_optim.param_groups:
            grp["weight_decay"] = 0.0
        if "torch_rng" in state and state["torch_rng"] is not None:
            torch.set_rng_state(state["torch_rng"].cpu())
        if "cuda_rng" in state and state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        step = int(state["step"])
        initial = int(state.get("initial_step", step))
        if step < initial:
            raise ValueError(f"checkpoint step ({step}) must be >= initial_step ({initial})")
        loaded_checkpoint_stage = state.get("stage_name")
        return step, initial

    start_step, initial_step = load_training_checkpoint(args.resume, ckpt)

    def build_stages():
        if args.stage_preset == "single":
            target_step = args.target_step
            if target_step <= start_step:
                raise ValueError(f"target_step ({target_step}) must be > start_step ({start_step})")
            return [{
                "name": "single",
                "resume_stage": None,
                "resume_step": None,
                "target_step": target_step,
                "muon_lr": args.muon_lr,
                "adam_lr": args.adam_lr,
                "min_lr_ratio": args.min_lr_ratio,
                "save_interval": args.save_interval,
                "eval_interval": args.eval_interval,
            }]

        if args.stage_preset == "adaptive":
            stages = [dict(s) for s in V17_ADAPTIVE_STAGES]
        else:
            stages = [dict(s) for s in V17_FRIEND_STAGES]
        names = [s["name"] for s in stages]
        if args.start_stage is not None:
            if args.start_stage not in names:
                raise ValueError(f"unknown start_stage={args.start_stage}; choices={names}")
            stages = stages[names.index(args.start_stage):]
        if args.stop_stage is not None:
            names = [s["name"] for s in stages]
            if args.stop_stage not in names:
                raise ValueError(f"unknown stop_stage={args.stop_stage}; choices={names}")
            stages = stages[:names.index(args.stop_stage) + 1]
        return stages

    stages = build_stages()

    if master_process:
        print(f"resumed v13 checkpoint from {args.resume} at step {start_step}")
        print(f"student params: {count_params(raw_model):,}")
        print(f"Muon  → {len(muon_params)} tensors, {sum(p.numel() for p in muon_params):,} params")
        print(f"AdamW → {len(adam_decay)+len(adam_nodecay)} tensors, weight_decay=0")
        print(
            f"stage_preset={args.stage_preset}, branch_mode={args.branch_mode}, "
            f"warmup={args.warmup_steps}, global_schedule={args.global_lr_schedule}"
        )
        for stage in stages:
            print(
                f"  stage {stage['name']}: target={stage['target_step']} "
                f"muon={stage['muon_lr']:.1e} adam={stage['adam_lr']:.1e} "
                f"min_ratio={stage['min_lr_ratio']} save={stage['save_interval']} eval={stage['eval_interval']}"
            )
        for s in train_loader.sources:
            print(f"  {s['name']}: weight={s['weight']:.2f}")

    # ── Logging / checkpoint helpers ───────────────────────────────────────────
    submission_config_fields = (
        "block_size", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd",
        "dropout", "bias", "norm_type", "position_embedding", "mlp_type",
        "intermediate_size", "tie_embeddings",
    )
    run_root = os.path.join(args.log_root, model_config.exp_name)
    os.makedirs(run_root, exist_ok=True)
    run_date = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(run_root, f"log_{run_date}_{model_config.exp_name}.txt")
    submission_template = os.path.join(os.path.dirname(__file__), "submission_model_template.py")

    with open(log_file, "a") as f:
        f.write(
            f"\n# v17 staged run from {args.resume} at step {start_step}, "
            f"initial_step {initial_step}, total_batch_size {total_batch_size}, "
            f"preset {args.stage_preset}, branch_mode {args.branch_mode}\n"
        )

    def stage_dir(stage_name):
        out = os.path.join(run_root, stage_name)
        os.makedirs(out, exist_ok=True)
        return out

    def train_ckpt_path(stage_name, step):
        return os.path.join(stage_dir(stage_name), f"train_ckpt_{step:05d}.pt")

    def best_meta_path(stage_name):
        return os.path.join(stage_dir(stage_name), "best_checkpoint.json")

    def read_best_checkpoint(stage_name):
        path = best_meta_path(stage_name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def record_stage_eval(stage_name, step, ckpt_path, results):
        if results is None:
            return
        meta = {
            "stage": stage_name,
            "step": int(step),
            "checkpoint": ckpt_path,
            "perplexity": float(results["perplexity"]),
            "avg_loss_nats": float(results["avg_loss_nats"]),
        }
        metrics_path = os.path.join(stage_dir(stage_name), "stage_metrics.jsonl")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(meta, sort_keys=True) + "\n")
        best = read_best_checkpoint(stage_name)
        if best is None or meta["perplexity"] < float(best["perplexity"]):
            tmp = best_meta_path(stage_name) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
            os.replace(tmp, best_meta_path(stage_name))

    def export_submission_bundle(model_obj, step, stage_name):
        """Export raw v13-continuation model weights."""
        out_dir = os.path.join(
            stage_dir(stage_name),
            f"submission_{run_date}_{model_config.exp_name}_{stage_name}_{step:05d}",
        )
        os.makedirs(out_dir, exist_ok=True)
        cfg = {k: getattr(model_obj.config, k) for k in submission_config_fields}
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        shutil.copyfile(submission_template, os.path.join(out_dir, "model.py"))
        torch.save(model_obj.state_dict(), os.path.join(out_dir, "checkpoint.pt"))
        return out_dir

    def save_train_checkpoint(step, stage_name, current_initial_step):
        ckpt_path = train_ckpt_path(stage_name, step)
        payload = {
            "model":        raw_model.state_dict(),
            "muon_optim":   muon_optim.state_dict(),
            "adam_optim":   adam_optim.state_dict(),
            "train_loader": train_loader.state_dict(),
            "torch_rng":    torch.get_rng_state(),
            "cuda_rng":     torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "config":       {k: getattr(raw_model.config, k) for k in submission_config_fields},
            "step":         step,
            "initial_step": current_initial_step,
            "stage_name":   stage_name,
        }
        torch.save(payload, ckpt_path)
        shutil.copyfile(ckpt_path, os.path.join(stage_dir(stage_name), "train_ckpt_latest.pt"))
        shutil.copyfile(ckpt_path, os.path.join(run_root, "train_ckpt_latest.pt"))
        return ckpt_path

    # ── Training loop ──────────────────────────────────────────────────────────
    vocab_size = raw_model.config.vocab_size

    current_step = start_step
    current_initial_step = initial_step

    for stage in stages:
        stage_name = stage["name"]
        target_step = int(stage["target_step"])
        stage_muon_lr = float(stage["muon_lr"])
        stage_adam_lr = float(stage["adam_lr"])
        stage_min_lr_ratio = float(stage["min_lr_ratio"])
        stage_save_interval = int(stage["save_interval"])
        stage_eval_interval = int(stage["eval_interval"])
        stage_lr_decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else target_step

        if current_step >= target_step:
            if master_process:
                print(f"skipping stage {stage_name}: current_step={current_step}, target={target_step}")
            continue

        resume_stage = stage.get("resume_stage")
        resume_step = stage.get("resume_step")
        if resume_stage is not None and resume_step is not None:
            expected_resume_step = int(resume_step)
            continuing_interrupted_stage = (
                loaded_checkpoint_stage == stage_name
                and expected_resume_step <= current_step < target_step
            )
            if continuing_interrupted_stage:
                if master_process:
                    print(f"\n=== stage {stage_name}: continuing interrupted checkpoint at {current_step} ===")
            elif args.branch_mode == "sequential":
                if master_process:
                    print(f"\n=== stage {stage_name}: sequential continue from current_step={current_step} ===")
            else:
                branch_path = None
                if args.branch_mode == "best":
                    if ddp:
                        dist.barrier()
                    best = read_best_checkpoint(resume_stage)
                    if best is not None and os.path.exists(best["checkpoint"]):
                        branch_path = best["checkpoint"]
                        if master_process:
                            print(
                                f"\n=== stage {stage_name}: best branch from {resume_stage} "
                                f"step {best['step']} ppl {best['perplexity']:.4f} ==="
                            )
                    elif master_process:
                        print(
                            f"\n=== stage {stage_name}: no best checkpoint found for {resume_stage}; "
                            f"falling back to CLI step {expected_resume_step} ==="
                        )
                if branch_path is None:
                    branch_path = train_ckpt_path(resume_stage, expected_resume_step)
                if ddp:
                    dist.barrier()
                if os.path.exists(branch_path):
                    current_step, current_initial_step = load_training_checkpoint(branch_path)
                    if master_process:
                        print(f"\n=== stage {stage_name}: loaded branch {branch_path} ===")
                elif current_step != expected_resume_step:
                    raise FileNotFoundError(
                        f"stage {stage_name} expects {branch_path}, but it does not exist "
                        f"and current_step={current_step} != resume_step={resume_step}"
                    )

        stage_start_step = current_step
        if master_process:
            print(
                f"\n=== stage {stage_name}: {stage_start_step} -> {target_step} | "
                f"muon {stage_muon_lr:.2e} adam {stage_adam_lr:.2e} "
                f"min_ratio {stage_min_lr_ratio} save {stage_save_interval} eval {stage_eval_interval} ==="
            )
            with open(log_file, "a") as f:
                f.write(
                    f"# stage {stage_name} start {stage_start_step} target {target_step} "
                    f"muon_lr {stage_muon_lr:.8e} adam_lr {stage_adam_lr:.8e} "
                    f"min_lr_ratio {stage_min_lr_ratio:.4f} save_interval {stage_save_interval} "
                    f"eval_interval {stage_eval_interval} lr_decay_steps {stage_lr_decay_steps}\n"
                )

        for step in range(stage_start_step + 1, target_step + 1):
            schedule_step = step if args.global_lr_schedule else step - stage_start_step - 1
            last_step = step == target_step
            t0 = time.time()

            train_model.train()
            muon_optim.zero_grad(set_to_none=True)
            adam_optim.zero_grad(set_to_none=True)

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

            if ddp:
                dist.all_reduce(ce_loss_accum, op=dist.ReduceOp.AVG)

            norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)

            muon_lr = get_lr(
                schedule_step, stage_muon_lr, args.warmup_steps,
                stage_lr_decay_steps, stage_min_lr_ratio,
            )
            adam_lr = get_lr(
                schedule_step, stage_adam_lr, args.warmup_steps,
                stage_lr_decay_steps, stage_min_lr_ratio,
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

            save_due = (
                step > stage_start_step
                and ((step - stage_start_step) % stage_save_interval == 0 or last_step)
            )
            eval_due = (
                step > stage_start_step
                and (step - stage_start_step) % stage_eval_interval == 0
                and not save_due
            )

            if master_process:
                print(
                    f"stage {stage_name} | step {step:5d} | ce {ce_loss_accum.item():.6f} | "
                    f"muon {muon_lr:.2e} adam {adam_lr:.2e} | "
                    f"norm {norm:.4f} | dt {dt*1000:.0f}ms | {tokens_per_sec:.0f} tok/s"
                )
                with open(log_file, "a") as f:
                    f.write(
                        f"{step} stage {stage_name} train ce {ce_loss_accum.item():.6f} "
                        f"muon_lr {muon_lr:.8e} adam_lr {adam_lr:.8e}\n"
                    )

                if eval_due:
                    quick = run_in_memory_eval(
                        raw_model, args.val_bin_path, device, amp_dtype,
                        batch_size=args.eval_batch_size,
                    )
                    print(f"quick_eval | ppl {quick['perplexity']:.4f} | loss {quick['avg_loss_nats']:.6f}")
                    with open(log_file, "a") as f:
                        f.write(
                            f"{step} stage {stage_name} quick_eval ppl {quick['perplexity']:.4f} "
                            f"loss {quick['avg_loss_nats']:.6f} "
                            f"tok {quick['total_tokens_evaluated']}\n"
                        )

                if save_due:
                    counts = train_loader.consume_source_counts()
                    msg = " ".join(f"{k}={v}" for k, v in counts.items())
                    ckpt_path = save_train_checkpoint(step, stage_name, current_initial_step)
                    export_dir = export_submission_bundle(raw_model, step, stage_name)
                    eval_error = None
                    try:
                        results, elapsed = run_official_eval(
                            export_dir, device, args.evaluate_py, args.val_bin_path
                        )
                    except subprocess.CalledProcessError as exc:
                        results, elapsed = None, 0.0
                        eval_error = f"official_eval_failed returncode={exc.returncode}"
                    with open(log_file, "a") as f:
                        f.write(f"{step} stage {stage_name} sampled_windows {msg}\n")
                        f.write(f"{step} stage {stage_name} train_ckpt {ckpt_path}\n")
                        f.write(f"{step} stage {stage_name} export {export_dir}\n")
                        if results is None:
                            f.write(f"{step} stage {stage_name} evaluate_failed {eval_error}\n")
                        else:
                            f.write(
                                f"{step} stage {stage_name} evaluate ppl {results['perplexity']:.4f} "
                                f"loss {results['avg_loss_nats']:.6f} "
                                f"tok {results['total_tokens_evaluated']} sec {elapsed:.1f}\n"
                            )
                    if results is not None:
                        record_stage_eval(stage_name, step, ckpt_path, results)
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

            current_step = step

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
