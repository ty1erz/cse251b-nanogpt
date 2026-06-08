"""Shared training loop for the single-source GPT baseline ablations."""

import inspect
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from single_source_loader import SingleSourceDataLoader


def _configure_optimizer(model, weight_decay, learning_rate, device_type):
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]
    no_decay = [p for p in params.values() if p.dim() < 2]
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused and device_type == "cuda",
    )


def train_baseline(
    *,
    experiment_name,
    source_name,
    data_dir,
    file_format,
    output_root,
    model_class,
    config_class,
):
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP baseline training requires CUDA")
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master = True
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    total_batch_size = 524288
    B = 16
    T = 1024
    max_steps = 5000
    warmup_steps = 715
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    if total_batch_size % (B * T * world_size) != 0:
        raise ValueError(
            "total batch size must be divisible by micro-batch tokens times "
            "the DDP world size"
        )
    grad_accum_steps = total_batch_size // (B * T * world_size)

    if master:
        print(f"=== {experiment_name} ===")
        print(f"device: {device}")
        print(f"data: {data_dir}")
        print(f"gradient accumulation steps: {grad_accum_steps}")

    train_loader = SingleSourceDataLoader(
        B, T, rank, world_size, "train", data_dir, file_format, source_name,
        master,
    )
    val_loader = SingleSourceDataLoader(
        B, T, rank, world_size, "val", data_dir, file_format, source_name,
        master,
    )

    torch.set_float32_matmul_precision("high")
    config = config_class(
        block_size=1024,
        vocab_size=50304,
        n_layer=8,
        n_head=8,
        n_embd=512,
    )
    raw_model = model_class(config).to(device)
    model = DDP(raw_model, device_ids=[local_rank]) if ddp else raw_model
    optimizer = _configure_optimizer(raw_model, 0.1, max_lr, device_type)

    def learning_rate(step):
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + coefficient * (max_lr - min_lr)

    log_dir = os.path.join(output_root, "logs", experiment_name)
    log_file = os.path.join(log_dir, "log.txt")
    if master:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "w", encoding="utf-8"):
            pass

    for step in range(max_steps):
        started = time.time()
        last_step = step == max_steps - 1

        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            val_loss = torch.zeros((), device=device)
            with torch.no_grad():
                for _ in range(20):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(
                        device_type=device_type, dtype=torch.bfloat16
                    ):
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), y.view(-1)
                        )
                    val_loss += loss.detach() / 20
            if ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            if master:
                print(f"validation loss: {val_loss.item():.4f}")
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"{step} val {val_loss.item():.4f}\n")

        model.train()
        optimizer.zero_grad()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1
                )
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            loss /= grad_accum_steps
            train_loss += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - started
        tokens_per_second = total_batch_size / elapsed
        if master:
            message = (
                f"step {step:5d} | loss: {train_loss.item():.6f} | "
                f"lr {lr:.4e} | norm: {grad_norm:.4f} | "
                f"dt: {elapsed * 1000:.2f}ms | tok/sec: {tokens_per_second:.2f}"
            )
            print(message)
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"{step} train {train_loss.item():.6f}\n")

        if master and last_step:
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss.item(),
                    "data_source": source_name,
                },
                os.path.join(log_dir, f"model_{step:05d}.pt"),
            )

    if ddp:
        destroy_process_group()
