"""
Phase 1 trainer for the CSE 251B NanoGPT competition.

  - Architecture:  model_v2.GPT (RoPE, RMSNorm, ReLU², QK-Norm, ~91M params)
  - Data:          mix_loader.MixedDataLoader (FineWeb-Edu / Wiki / Sci / Books
                   at 50/20/15/15)
  - Eval:          val loss on FineWeb-Edu val shard during training; final
                   val.bin perplexity computation if val.bin is reachable.
  - Output:        log_v2/<RUN_NAME>/
                       model_<step>.pt        # checkpoint (model + config + step)
                       MODEL_CARD.md          # human-readable summary
                       train_log.txt          # per-step train / val numbers

Launch:
    python train_v2.py
    python train_v2.py --run_name v2_baseline_5k --max_steps 5000
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

# local imports — must work whether run as `python train_v2.py` from this dir
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_v2 import GPT, GPTConfigV2, num_params
from mix_loader import MixedDataLoader


# ---------------------------------------------------------------------------
# Config (edit defaults here, or override on CLI)

DEFAULTS = dict(
    run_name=None,                 # auto-generated if not provided
    log_root=os.path.join(HERE, "log_v2"),
    # data
    total_batch_size=524288,       # tokens per optimizer step (GPT-3 small recipe)
    micro_batch=16,                # per-step micro batch; must divide 512 (T=1024)
    seq_len=1024,
    # schedule
    max_steps=5000,                # 5000 × 524288 ≈ 2.6B tokens
    warmup_steps=200,              # modded-nanogpt finding
    # optim
    max_lr=6e-4,
    min_lr_ratio=0.1,              # terminal LR = min_lr_ratio * max_lr (NOT 0)
    weight_decay=0.1,
    grad_clip=1.0,
    # eval cadence
    eval_every=250,
    eval_iters=20,                 # number of val_loader micro-batches
    ckpt_every=2500,               # checkpoint cadence
    # bookkeeping
    seed=1337,
    val_bin_path=os.path.abspath(os.path.join(HERE, "..", "cse251b-nanogpt", "val.bin")),
)


def parse_args():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", action=argparse.BooleanOptionalAction, default=v)
        elif v is None:
            ap.add_argument(f"--{k}", type=str, default=None)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# val.bin perplexity (matches evaluate.py, run once at end)

@torch.no_grad()
def perplexity_on_valbin(model, val_path, device, block_size=1024, batch_size=8):
    if not os.path.exists(val_path):
        return None
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    data = torch.from_numpy(data.astype(np.int64))
    n_chunks = (len(data) - 1) // block_size
    n_chunks = (n_chunks // batch_size) * batch_size
    if n_chunks == 0:
        return None
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, n_chunks, batch_size):
        x = torch.stack([data[j*block_size : j*block_size+block_size] for j in range(i, i+batch_size)]).to(device)
        y = torch.stack([data[j*block_size+1 : j*block_size+block_size+1] for j in range(i, i+batch_size)]).to(device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            logits, _ = model(x)
        # match evaluator's slice
        logits = logits[:, :, :50257]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return dict(perplexity=math.exp(avg_loss), avg_loss_nats=avg_loss, tokens=total_tokens)


# ---------------------------------------------------------------------------
# Model card writer

def write_model_card(out_dir, args, cfg, n_params, started_at, ended_at,
                     final_step, final_val_loss, valbin_result, mix):
    elapsed_h = (ended_at - started_at) / 3600.0
    card = f"""# Model card — {args.run_name}

**Generated:** {datetime.datetime.fromtimestamp(ended_at).isoformat(timespec="seconds")}

## Architecture
- Class: `model_v2.GPT` ([model_v2.py](../../model_v2.py))
- Config: `n_layer={cfg.n_layer}`, `n_head={cfg.n_head}`, `n_embd={cfg.n_embd}`, `block_size={cfg.block_size}`, `vocab_size={cfg.vocab_size}`
- **Parameters:** {n_params:,}  ({n_params/1e6:.2f} M)
- Modern components: RoPE positional encoding, RMSNorm, ReLU² MLP, QK-Norm, tied embeddings, bias-free linears

## Training data
- Mix: {", ".join(f"{k}={v:.0%}" for k, v in mix.items())}
- Loader: `MixedDataLoader` ([mix_loader.py](../../mix_loader.py))

## Hyperparameters
- micro_batch={args.micro_batch}, seq_len={args.seq_len}, total_batch_size={args.total_batch_size}
- max_steps={args.max_steps} (≈ {args.max_steps * args.total_batch_size / 1e9:.2f} B tokens)
- max_lr={args.max_lr}, warmup_steps={args.warmup_steps}, min_lr_ratio={args.min_lr_ratio}
- weight_decay={args.weight_decay}, grad_clip={args.grad_clip}
- optimizer: AdamW (fused), betas=(0.9, 0.95)

## Results
- Final step: {final_step}
- **Final val loss (FineWeb-Edu val shard):** {final_val_loss:.4f}
"""
    if valbin_result is not None:
        card += (
            f"- **val.bin perplexity:** {valbin_result['perplexity']:.4f}\n"
            f"- val.bin avg loss (nats): {valbin_result['avg_loss_nats']:.4f}\n"
            f"- tokens evaluated: {valbin_result['tokens']:,}\n"
        )
    else:
        card += "- val.bin perplexity: (not run; val.bin not found)\n"

    card += f"""
## Reproducibility
- Started: {datetime.datetime.fromtimestamp(started_at).isoformat(timespec="seconds")}
- Ended:   {datetime.datetime.fromtimestamp(ended_at).isoformat(timespec="seconds")}
- Wall-clock: {elapsed_h:.2f} h
- Seed: {args.seed}
- Run script: `train_v2.py --run_name {args.run_name} --max_steps {args.max_steps}`

Checkpoints in this directory: every {args.ckpt_every} steps + final.
"""
    with open(os.path.join(out_dir, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
        f.write(card)


# ---------------------------------------------------------------------------
# Train

def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"v2_arch+mix_{args.max_steps}st_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

    out_dir = os.path.join(args.log_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # DDP setup (single-GPU friendly)
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(torch.backends, "mps") and not torch.cuda.is_available() and torch.backends.mps.is_available():
            device = "mps"

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    if master:
        print(f"=== run_name: {args.run_name} ===")
        print(f"out_dir: {out_dir}")
        print(f"device: {device}")

    # Seeding
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Batch math
    B, T = args.micro_batch, args.seq_len
    assert args.total_batch_size % (B * T * ddp_world_size) == 0, \
        f"total_batch_size {args.total_batch_size} not divisible by B*T*world {B*T*ddp_world_size}"
    grad_accum_steps = args.total_batch_size // (B * T * ddp_world_size)
    if master:
        print(f"total_batch_size = {args.total_batch_size}")
        print(f"grad_accum_steps = {grad_accum_steps}")

    # Data
    train_loader = MixedDataLoader(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
        split="train", master_process=master,
    )
    val_loader = MixedDataLoader(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
        split="val", master_process=master,
    )

    torch.set_float32_matmul_precision("high")

    # Model
    cfg = GPTConfigV2()
    model = GPT(cfg).to(device)
    n_params = num_params(model)
    if master:
        print(f"params: {n_params:,}  ({n_params/1e6:.2f} M)")
        assert n_params < 100_000_000, "OVER 100M PARAM CAP"

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Optimizer + LR schedule
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.max_lr,
        device_type=device_type, master=master,
    )
    min_lr = args.max_lr * args.min_lr_ratio

    def get_lr(step):
        if step < args.warmup_steps:
            return args.max_lr * (step + 1) / args.warmup_steps
        if step >= args.max_steps:
            return min_lr
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + coeff * (args.max_lr - min_lr)

    # Logging
    log_path = os.path.join(out_dir, "train_log.txt")
    with open(log_path, "w") as f:
        f.write(f"# {args.run_name}\n# started {datetime.datetime.now().isoformat()}\n")

    final_val_loss = float("inf")
    started_at = time.time()

    for step in range(args.max_steps):
        t0 = time.time()
        last_step = (step == args.max_steps - 1)

        # ---- val pass (cheap, frequent) ----
        if step % args.eval_every == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                vacc = 0.0
                for _ in range(args.eval_iters):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    vacc += loss.detach() / args.eval_iters
            if ddp:
                dist.all_reduce(vacc, op=dist.ReduceOp.AVG)
            final_val_loss = vacc.item()
            if master:
                print(f"step {step:5d} | val {final_val_loss:.4f}")
                with open(log_path, "a") as f:
                    f.write(f"{step} val {final_val_loss:.6f}\n")

        # ---- checkpoint ----
        if master and step > 0 and (step % args.ckpt_every == 0 or last_step):
            ckpt = {
                "model": raw_model.state_dict(),
                "config": cfg,
                "step": step,
                "val_loss": final_val_loss,
                "n_params": n_params,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(out_dir, f"model_{step:06d}.pt"))

        # ---- train step ----
        model.train()
        optimizer.zero_grad()
        loss_acc = 0.0
        for micro in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_acc += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        toks = B * T * grad_accum_steps * ddp_world_size
        if master:
            print(
                f"step {step:5d} | train {loss_acc.item():.4f} | "
                f"lr {lr:.3e} | norm {norm:.3f} | "
                f"dt {dt*1000:.0f}ms | tok/s {toks/dt:,.0f}"
            )
            with open(log_path, "a") as f:
                f.write(f"{step} train {loss_acc.item():.6f}\n")

    ended_at = time.time()

    # ---- one-shot val.bin perplexity (the metric we actually care about) ----
    valbin_result = None
    if master:
        print("\nrunning val.bin perplexity...")
        valbin_result = perplexity_on_valbin(raw_model, args.val_bin_path, device)
        if valbin_result:
            print(f"val.bin perplexity: {valbin_result['perplexity']:.4f}")
        else:
            print("val.bin not found, skipping")

    # ---- model card ----
    if master:
        write_model_card(
            out_dir, args, cfg, n_params,
            started_at, ended_at,
            final_step=args.max_steps - 1,
            final_val_loss=final_val_loss,
            valbin_result=valbin_result,
            mix=train_loader.weights,
        )
        print(f"\nwrote {os.path.join(out_dir, 'MODEL_CARD.md')}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
