#!/usr/bin/env python3
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
from model import GPT, GPTConfig


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PUBLIC_VAL = os.environ.get("VAL_BIN", os.path.join(ROOT_DIR, "val.bin"))
DEFAULT_EVAL_PY = os.path.join(ROOT_DIR, "evaluate.py")
LOG_ROOT = os.environ.get("LOG_ROOT", os.path.join(ROOT_DIR, "log"))
START_STEP = 0

FIRST_EPOCH_SOURCE_SPECS = [
    {
        "name": "fineweb",
        "path": os.environ.get(
            "FINEWEB_PATH",
            "/data/fengfei/cse251b-nanogpt/build-nanogpt/tokenized_sources/fineweb_full",
        ),
        "weight": 0.50,
    },
    {
        "name": "wikipedia",
        "path": os.environ.get(
            "WIKIPEDIA_PATH",
            "/data/fengfei/cse251b-nanogpt-zzv-train/data/wikipedia",
        ),
        "weight": 0.20,
    },
    {
        "name": "science",
        "path": os.environ.get(
            "SCIENCE_PATH",
            "/data/fengfei/cse251b-nanogpt-zzv-train/data/science",
        ),
        "weight": 0.15,
    },
    {
        "name": "books",
        "path": os.environ.get(
            "BOOKS_PATH",
            "/data/fengfei/cse251b-nanogpt-zzv-train/data/books",
        ),
        "weight": 0.15,
    },
]

MICRO_BATCH = 16
SEQ_LEN = 1024
TOTAL_BATCH_SIZE = 524288
TARGET_STEP = 38000
LR_DECAY_STEPS = 38000
WARMUP_STEPS = 200
SAVE_INTERVAL = 2000
EVAL_INTERVAL = 2000
MUON_LR = 1.3e-2
MUON_MOMENTUM = 0.95
ADAM_LR = 5.2e-4
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0


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


def configure_optimizers(model, device_type):
    muon_params, adam_decay, adam_nodecay = split_params_for_muon(model)
    muon_optim = Muon(
        muon_params,
        lr=MUON_LR,
        momentum=MUON_MOMENTUM,
        nesterov=True,
        ns_steps=5,
        weight_decay=WEIGHT_DECAY,
    )
    adam_optim = torch.optim.AdamW(
        [
            {"params": adam_decay, "weight_decay": 0.0},
            {"params": adam_nodecay, "weight_decay": 0.0},
        ],
        lr=ADAM_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=(device_type == "cuda"),
    )
    return muon_optim, adam_optim, muon_params, adam_decay, adam_nodecay


def init_fresh_model(model):
    """nanoGPT-style initialization for fresh training.

    model.py does not define custom init, and PyTorch's
    default Embedding std is far too large for GPT logits at step 0.
    """
    n_layer = model.config.n_layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            std = 0.02
            if name.endswith("attn.c_proj") or name.endswith("mlp.down_proj") or name.endswith("mlp.c_proj"):
                std *= (2 * n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def load_submission_model(submission_dir, device):
    config_path = os.path.join(submission_dir, "config.json")
    ckpt_path = os.path.join(submission_dir, "checkpoint.pt")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = GPTConfig(**json.load(f))
    model = GPT(cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model, cfg, ckpt_path


class MultiSourceWindowLoader:
    def __init__(self, source_specs, B, T, process_rank, num_processes, seed=2025, pin_memory=False):
        self.B = B
        self.T = T
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
                "name": spec["name"],
                "path": path,
                "weight": float(spec["weight"]),
                "shard_i": 0,
                "pos": self.T * self.process_rank,
                "epochs": 0,
            }
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "train.bin")):
                tokens = np.memmap(os.path.join(path, "train.bin"), dtype=np.uint16, mode="r")
                start = int(spec.get("start", 0))
                end = int(spec.get("end", len(tokens)))
                if end > len(tokens):
                    raise ValueError(f"Source {spec['name']} end={end} exceeds file length {len(tokens)}")
                if end - start < (T + 1):
                    raise ValueError(f"Source {spec['name']} too short for block_size={T}")
                source.update({"mode": "train_bin", "shards": [tokens], "starts": [start], "ends": [end]})
            elif os.path.isdir(path):
                shard_paths = sorted(
                    os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')
                )
                if not shard_paths:
                    missing.append((spec["name"], path + ' (no .npy shards or train.bin)'))
                    continue
                shard_tokens = [np.load(sp, mmap_mode='r') for sp in shard_paths]
                for arr in shard_tokens:
                    if len(arr) < (T + 1):
                        raise ValueError(f"Source {spec['name']} has shard shorter than block_size={T}")
                source.update({
                    "mode": "npy_shards",
                    "shards": shard_tokens,
                    "shard_paths": shard_paths,
                    "starts": [0] * len(shard_tokens),
                    "ends": [len(arr) for arr in shard_tokens],
                })
            else:
                tokens = np.memmap(path, dtype=np.uint16, mode="r")
                start = int(spec.get("start", 0))
                end = int(spec.get("end", len(tokens)))
                if end > len(tokens):
                    raise ValueError(f"Source {spec['name']} end={end} exceeds file length {len(tokens)}")
                if end - start < (T + 1):
                    raise ValueError(f"Source {spec['name']} too short for block_size={T}")
                source.update({"mode": "train_bin", "shards": [tokens], "starts": [start], "ends": [end]})

            self.sources.append(source)
            self.source_counts[spec["name"]] = 0

        if missing:
            msg = "\n".join(f"  - {name}: {path}" for name, path in missing)
            raise FileNotFoundError("Missing tokenized sources for the first epoch:\n" f"{msg}")

        weights = np.asarray([s["weight"] for s in self.sources], dtype=np.float64)
        self.weights = weights / weights.sum()
        self._build_source_cycle()

    def _build_source_cycle(self):
        # The 50/20/15/15 mix fits exactly in a 20-sample cycle.
        counts = np.rint(self.weights * 20).astype(int)
        while counts.sum() < 20:
            counts[np.argmax(self.weights * 20 - counts)] += 1
        while counts.sum() > 20:
            counts[np.argmax(counts - self.weights * 20)] -= 1
        cycle = []
        for source_idx, count in enumerate(counts):
            cycle.extend([source_idx] * int(count))
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
                return np.asarray(shard[start : start + self.T + 1], dtype=np.int64)

            source["shard_i"] += 1
            if source["shard_i"] >= len(source["shards"]):
                source["shard_i"] = 0
                source["epochs"] += 1
            source["pos"] = source["starts"][source["shard_i"]] + self.T * self.process_rank

    def consume_source_counts(self):
        counts = dict(self.source_counts)
        for key in self.source_counts:
            self.source_counts[key] = 0
        return counts

    def next_batch(self):
        x = torch.empty((self.B, self.T), dtype=torch.long, pin_memory=self.pin_memory)
        y = torch.empty((self.B, self.T), dtype=torch.long, pin_memory=self.pin_memory)
        for i in range(self.B):
            source_idx = self._next_source_idx()
            source = self.sources[int(source_idx)]
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
            "sources": {
                s["name"]: {
                    "shard_i": int(s["shard_i"]),
                    "pos": int(s["pos"]),
                    "epochs": int(s["epochs"]),
                }
                for s in self.sources
            },
            "source_counts": dict(self.source_counts),
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        saved_num_processes = int(state.get("num_processes", self.num_processes))
        if saved_num_processes != self.num_processes:
            raise ValueError(
                f"Cannot resume non-overlap loader with num_processes={self.num_processes}; "
                f"checkpoint used num_processes={saved_num_processes}."
            )
        self.source_cursor = int(state["source_cursor"])
        self.source_cycle = list(state["source_cycle"])
        saved_sources = state.get("sources", {})
        rank_delta = (self.process_rank - int(state.get("process_rank", 0))) * self.T
        for source in self.sources:
            if source["name"] in saved_sources:
                saved = saved_sources[source["name"]]
                source["shard_i"] = int(saved["shard_i"])
                source["pos"] = int(saved["pos"]) + rank_delta
                source["epochs"] = int(saved.get("epochs", 0))
        self.source_counts.update(state.get("source_counts", {}))


def run_official_eval(model_dir, device, evaluate_py, public_val_bin):
    output_json = os.path.join(model_dir, "evaluate_results.json")
    cmd = [
        sys.executable,
        evaluate_py,
        "--model_dir", model_dir,
        "--data", public_val_bin,
        "--device", device,
        "--output_json", output_json,
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    with open(output_json, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results, elapsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a first-epoch train_ckpt_*.pt checkpoint",
    )
    parser.add_argument("--target_step", type=int, default=TARGET_STEP, help="Stop after this global step")
    parser.add_argument("--micro_batch", type=int, default=MICRO_BATCH, help="Per-GPU micro batch size")
    parser.add_argument("--total_batch_size", type=int, default=TOTAL_BATCH_SIZE, help="Global token batch size")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the model")
    return parser.parse_args()


def main():
    args = parse_args()
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2025)
    torch.set_float32_matmul_precision("high")

    stage_cfg = {"start_step": START_STEP, "sources": FIRST_EPOCH_SOURCE_SPECS}

    B = args.micro_batch
    T = SEQ_LEN
    total_batch_size = args.total_batch_size
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    target_step = args.target_step
    warmup_steps = WARMUP_STEPS
    lr_decay_steps = LR_DECAY_STEPS
    eval_interval = EVAL_INTERVAL
    save_interval = SAVE_INTERVAL
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> gradient accumulation steps: {grad_accum_steps}")

    train_loader = MultiSourceWindowLoader(
        source_specs=stage_cfg["sources"],
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        seed=2025,
        pin_memory=(device_type == "cuda"),
    )

    start_step = stage_cfg["start_step"]
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model_config = GPTConfig(**cfg) if isinstance(cfg, dict) else GPTConfig()
        model = GPT(model_config)
        model.load_state_dict(ckpt["model"], strict=True)
        train_loader.load_state_dict(ckpt["train_loader"])
        start_step = int(ckpt["step"])
    else:
        model_config = GPTConfig()
        model = GPT(model_config)
        init_fresh_model(model)
    model.to(device)
    model_config.exp_name = "first_epoch"
    raw_model = model
    if args.compile and ddp:
        raise ValueError("--compile is only wired for single-GPU training in this script")
    train_model = torch.compile(raw_model) if args.compile else raw_model

    if ddp:
        train_model = DDP(train_model, device_ids=[ddp_local_rank])
        raw_model = train_model.module
    muon_optim, adam_optim, muon_params, adam_decay, adam_nodecay = configure_optimizers(
        raw_model, device_type
    )
    if args.resume:
        muon_optim.load_state_dict(ckpt["muon_optim"])
        adam_optim.load_state_dict(ckpt["adam_optim"])
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].detach().cpu())
        if "cuda_rng" in ckpt and ckpt["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["cuda_rng"].detach().cpu())
    amp_dtype = torch.bfloat16 if device_type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device_type == "cuda" else torch.float32
    )

    if master_process:
        if args.resume:
            print(f"init: resumed full training checkpoint {args.resume}")
        else:
            print("init: fresh GPTConfig() model with nanoGPT-style std=0.02 init, no resume checkpoint")
        print(f"start step: {start_step}")
        print(f"micro_batch: {B}")
        print(f"student params: {count_params(raw_model):,}")
        print(f"Muon  -> {len(muon_params)} tensors, {sum(p.numel() for p in muon_params):,} params")
        print(
            "AdamW -> "
            f"{len(adam_decay) + len(adam_nodecay)} tensors, "
            f"{sum(p.numel() for p in adam_decay) + sum(p.numel() for p in adam_nodecay):,} params"
        )
        print("source weights:")
        for s in train_loader.sources:
            print(f"  {s['name']}: weight={s['weight']:.2f} path={s['path']}")
        print(
            f"training: muon_lr={MUON_LR:.2e} adam_lr={ADAM_LR:.2e} min_lr_ratio={MIN_LR_RATIO} "
            f"warmup_steps={warmup_steps} target_step={target_step} lr_decay_steps={lr_decay_steps} "
            f"eval_interval={eval_interval} save_interval={save_interval}"
        )
        print("loss: hard CE only (no teacher, no KD)")

    muon_min_lr = MUON_LR * MIN_LR_RATIO
    adam_min_lr = ADAM_LR * MIN_LR_RATIO

    def get_lr(local_step, peak, floor):
        if local_step < warmup_steps:
            return peak * (local_step + 1) / warmup_steps
        if local_step > lr_decay_steps:
            return floor
        decay_ratio = (local_step - warmup_steps) / max(1, (lr_decay_steps - warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return floor + coeff * (peak - floor)

    log_dir = os.path.join(LOG_ROOT, "first_epoch")
    os.makedirs(log_dir, exist_ok=True)
    run_date = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"log_{run_date}_{model_config.exp_name}.txt")
    with open(log_file, "a" if args.resume else "w", encoding="utf-8") as f:
        if args.resume:
            f.write(f"# resumed from {args.resume} at step {start_step}, target_step {target_step}\n")

    submission_template = os.path.join(ROOT_DIR, "model.py")
    submission_config_fields = (
        "block_size", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd",
        "dropout", "bias", "norm_type", "position_embedding", "mlp_type",
        "intermediate_size", "tie_embeddings",
    )

    def export_submission_bundle(model_obj, step):
        export_dir = os.path.join(log_dir, f"submission_{run_date}_{model_config.exp_name}_{step:05d}")
        os.makedirs(export_dir, exist_ok=True)
        cfg = {k: getattr(model_obj.config, k) for k in submission_config_fields}
        with open(os.path.join(export_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        shutil.copyfile(submission_template, os.path.join(export_dir, "model.py"))
        torch.save(model_obj.state_dict(), os.path.join(export_dir, "checkpoint.pt"))
        return export_dir

    def save_train_checkpoint(model_obj, step):
        ckpt_path = os.path.join(log_dir, f"train_ckpt_{step:05d}.pt")
        ckpt = {
            "model": model_obj.state_dict(),
            "muon_optim": muon_optim.state_dict(),
            "adam_optim": adam_optim.state_dict(),
            "train_loader": train_loader.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "config": {k: getattr(model_obj.config, k) for k in submission_config_fields},
            "step": step,
        }
        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, os.path.join(log_dir, "train_ckpt_latest.pt"))
        return ckpt_path

    for step in range(start_step + 1, target_step + 1):
        local_step = step - 1
        last_step = step == target_step
        t0 = time.time()

        train_model.train()
        muon_optim.zero_grad(set_to_none=True)
        adam_optim.zero_grad(set_to_none=True)
        loss_accum = 0.0
        hard_loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if ddp:
                train_model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                student_logits = train_model(x)
                hard_loss = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), y.reshape(-1))
                loss = hard_loss
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            hard_loss_accum += hard_loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(hard_loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)
        muon_lr_now = get_lr(local_step, MUON_LR, muon_min_lr)
        adam_lr_now = get_lr(local_step, ADAM_LR, adam_min_lr)
        for param_group in muon_optim.param_groups:
            param_group["lr"] = muon_lr_now
        for param_group in adam_optim.param_groups:
            param_group["lr"] = adam_lr_now
        muon_optim.step()
        adam_optim.step()
        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | hard: {hard_loss_accum.item() / grad_accum_steps:.6f} | "
                f"muon {muon_lr_now:.4e} adam {adam_lr_now:.4e} | norm: {norm:.4f} | "
                f"dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{step} train {loss_accum.item():.6f} hard {hard_loss_accum.item() / grad_accum_steps:.6f} "
                    f"muon_lr {muon_lr_now:.8e} adam_lr {adam_lr_now:.8e}\n"
                )

            if step > stage_cfg["start_step"] and step % eval_interval == 0:
                counts = train_loader.consume_source_counts()
                msg = " ".join(f"{k}={v}" for k, v in counts.items())
                print(f"sampled_windows {msg}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} sampled_windows {msg}\n")

            if step > 0 and ((step - stage_cfg["start_step"]) % save_interval == 0 or last_step):
                train_ckpt_path = save_train_checkpoint(raw_model, step)
                export_dir = export_submission_bundle(raw_model, step)
                eval_results, eval_elapsed = run_official_eval(export_dir, device, DEFAULT_EVAL_PY, DEFAULT_PUBLIC_VAL)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} train_ckpt {train_ckpt_path}\n")
                    f.write(f"{step} export {export_dir}\n")
                    f.write(
                        f"{step} evaluate ppl {eval_results['perplexity']:.4f} loss {eval_results['avg_loss_nats']:.6f} "
                        f"tok {eval_results['total_tokens_evaluated']} sec {eval_elapsed:.2f}\n"
                    )
                print(
                    f"evaluate.py val.bin | ppl {eval_results['perplexity']:.4f} | "
                    f"loss {eval_results['avg_loss_nats']:.6f} | tokens {eval_results['total_tokens_evaluated']:,} | "
                    f"time {eval_elapsed:.1f}s"
                )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
