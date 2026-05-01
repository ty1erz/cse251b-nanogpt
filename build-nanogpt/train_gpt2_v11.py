#!/usr/bin/env python3
import inspect
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

from submission_model_template import GPT, GPTConfig


DEFAULT_SOURCE_CONFIG = "/data/fengfei/cse251b-nanogpt/build-nanogpt/data_configs/v8_multisource_default.json"
DEFAULT_PUBLIC_VAL = "/data/fengfei/cse251b-nanogpt/val.bin"
DEFAULT_EVAL_PY = "/data/fengfei/cse251b-nanogpt/evaluate.py"
DEFAULT_RESUME_SUBMISSION_DIR = "/data/fengfei/cse251b-nanogpt/build-nanogpt/log/v4/submission_20260426_alvin_v4_26000"
DEFAULT_RESUME_STEP = 26000


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


def configure_optimizers(model, weight_decay, learning_rate, device_type, beta1=0.9, beta2=0.95):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    return torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=1e-8,
        fused=use_fused,
    )


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
    def __init__(self, source_specs, B, T, process_rank, num_processes, seed=2025):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.rng = np.random.default_rng(seed + process_rank)
        self.sources = []
        self.source_counts = {}

        missing = []
        for spec in source_specs:
            path = spec["path"]
            if not os.path.exists(path):
                missing.append((spec["name"], path))
                continue
            tokens = np.memmap(path, dtype=np.uint16, mode="r")
            start = int(spec.get("start", 0))
            end = int(spec.get("end", len(tokens)))
            if end > len(tokens):
                raise ValueError(f"Source {spec['name']} end={end} exceeds file length {len(tokens)}")
            if end - start < (T + 1):
                raise ValueError(f"Source {spec['name']} too short for block_size={T}")
            self.sources.append(
                {
                    "name": spec["name"],
                    "path": path,
                    "tokens": tokens,
                    "start": start,
                    "end": end,
                    "weight": float(spec["weight"]),
                }
            )
            self.source_counts[spec["name"]] = 0

        if missing:
            msg = "\n".join(f"  - {name}: {path}" for name, path in missing)
            raise FileNotFoundError(
                "Missing tokenized sources for v10:\n"
                f"{msg}\n"
                "Create/edit those .bin paths in data_configs/v8_multisource_default.json first."
            )

        weights = np.asarray([s["weight"] for s in self.sources], dtype=np.float64)
        self.weights = weights / weights.sum()

    def consume_source_counts(self):
        counts = dict(self.source_counts)
        for key in self.source_counts:
            self.source_counts[key] = 0
        return counts

    def next_batch(self):
        x = torch.empty((self.B, self.T), dtype=torch.long)
        y = torch.empty((self.B, self.T), dtype=torch.long)
        choices = self.rng.choice(len(self.sources), size=self.B, p=self.weights)
        for i, source_idx in enumerate(choices):
            source = self.sources[int(source_idx)]
            low = source["start"]
            high = source["end"] - (self.T + 1)
            start = int(self.rng.integers(low, high + 1))
            buf = np.asarray(source["tokens"][start : start + self.T + 1], dtype=np.int64)
            x[i] = torch.from_numpy(buf[:-1].copy())
            y[i] = torch.from_numpy(buf[1:].copy())
            self.source_counts[source["name"]] += 1
        return x, y


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


def main():
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

    with open(DEFAULT_SOURCE_CONFIG, "r", encoding="utf-8") as f:
        stage_cfg = json.load(f)
    stage_cfg["resume_submission_dir"] = DEFAULT_RESUME_SUBMISSION_DIR
    stage_cfg["resume_step"] = DEFAULT_RESUME_STEP
    if "sources" in stage_cfg:
        for source in stage_cfg["sources"]:
            source["weight"] = 0.25

    B = 8
    T = 1024
    total_batch_size = 262144
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    max_lr = 4e-4
    min_lr = 4e-5
    warmup_steps = 100
    train_steps = 5000
    eval_interval = 500
    save_interval = 500
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
    )

    model, model_config, ckpt_path = load_submission_model(stage_cfg["resume_submission_dir"], device)
    model_config.exp_name = "alvin_v11"

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    optimizer = configure_optimizers(raw_model, 0.1, max_lr, device_type, 0.9, 0.95)
    amp_dtype = torch.bfloat16 if device_type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device_type == "cuda" else torch.float32
    )

    if master_process:
        print(f"resume checkpoint: {ckpt_path}")
        print(f"resume step: {stage_cfg['resume_step']}")
        print(f"student params: {count_params(raw_model):,}")
        print("source weights:")
        for s in train_loader.sources:
            print(f"  {s['name']}: weight={s['weight']:.2f} path={s['path']}")
        print(
            f"training: max_lr={max_lr:.2e} min_lr={min_lr:.2e} warmup_steps={warmup_steps} "
            f"train_steps={train_steps} eval_interval={eval_interval} save_interval={save_interval}"
        )
        print("loss: hard CE only (no teacher, no KD)")

    def get_lr(local_step):
        if local_step < warmup_steps:
            return max_lr * (local_step + 1) / warmup_steps
        if local_step > train_steps:
            return min_lr
        decay_ratio = (local_step - warmup_steps) / max(1, (train_steps - warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    log_dir = os.path.join("log", "v11")
    os.makedirs(log_dir, exist_ok=True)
    run_date = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"log_{run_date}_{model_config.exp_name}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        pass

    submission_template = os.path.join(os.path.dirname(__file__), "submission_model_template.py")
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

    for local_step in range(train_steps):
        step = stage_cfg["resume_step"] + local_step
        last_step = local_step == train_steps - 1
        t0 = time.time()

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        hard_loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                student_logits = model(x)
                hard_loss = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), y.reshape(-1))
                loss = hard_loss
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            hard_loss_accum += hard_loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(hard_loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(local_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | hard: {hard_loss_accum.item() / grad_accum_steps:.6f} | "
                f"lr {lr:.4e} | norm: {norm:.4f} | "
                f"dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{step} train {loss_accum.item():.6f} hard {hard_loss_accum.item() / grad_accum_steps:.6f}\n"
                )

            if step > stage_cfg["resume_step"] and step % eval_interval == 0:
                counts = train_loader.consume_source_counts()
                msg = " ".join(f"{k}={v}" for k, v in counts.items())
                print(f"sampled_windows {msg}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} sampled_windows {msg}\n")

            if step > 0 and ((step - stage_cfg["resume_step"]) % save_interval == 0 or last_step):
                export_dir = export_submission_bundle(raw_model, step)
                eval_results, eval_elapsed = run_official_eval(export_dir, device, DEFAULT_EVAL_PY, DEFAULT_PUBLIC_VAL)
                with open(log_file, "a", encoding="utf-8") as f:
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
