"""
Phase 2 trainer (v3) for the CSE 251B NanoGPT competition.

  Differences vs train_v2.py:
    * Optimizer split: Muon for 2-D hidden weights + AdamW for embeddings/norms.
    * Mix ratio shifted to FineWeb-heavy 59/17/12/12 (vs 50/20/15/15 in v2).
    * Default schedule: 5100 steps, checkpoint every 1700 steps (3 saves).
    * WD 0 on embeddings (v2 had a bug applying 0.1 there).

  Architecture: imported from model_v2 (same RoPE / RMSNorm / ReLU² / QK-Norm,
  ~91M params). No model_v3.py needed — the architecture didn't change.

Launch:
    python train_v3.py --run_name v3_muon_5100
"""

import argparse
import datetime
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_v2 import GPT, GPTConfigV2, num_params  # noqa: E402
from mix_loader import MixedDataLoader  # noqa: E402
from muon import Muon, split_params_for_muon  # noqa: E402


# ---------------------------------------------------------------------------
# Config

V3_MIX = {
    "fineweb": 0.59,
    "wikipedia": 0.17,
    "science": 0.12,
    "books": 0.12,
}

DEFAULTS = dict(
    run_name=None,
    log_root=os.path.join(HERE, "log_v3"),
    resume=None,                  # path to a v3 checkpoint to resume from
    # data
    total_batch_size=524288,
    micro_batch=16,
    seq_len=1024,
    # schedule
    max_steps=5100,
    warmup_steps=200,
    ckpt_every=1700,
    eval_every=250,
    eval_iters=20,
    # optim — Muon hidden weights, AdamW embed / norms
    muon_lr=2e-2,
    muon_momentum=0.95,
    adam_lr=8e-4,
    min_lr_ratio=0.1,
    weight_decay=0.1,            # applied to Muon hidden only; embeddings + norms get 0
    grad_clip=1.0,
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
# val.bin perplexity (matches evaluate.py)

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
        logits = logits[:, :, :50257]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return dict(perplexity=math.exp(avg_loss), avg_loss_nats=avg_loss, tokens=total_tokens)


# ---------------------------------------------------------------------------
# Model card writer

def write_model_card(out_dir, args, cfg, n_params, started_at, ended_at,
                     final_step, final_val_loss, valbin_result, mix,
                     n_muon_params, n_adam_params):
    elapsed_h = (ended_at - started_at) / 3600.0
    card = f"""# Model card — {args.run_name}

**Generated:** {datetime.datetime.fromtimestamp(ended_at).isoformat(timespec="seconds")}

## Architecture
- Class: `model_v2.GPT` ([model_v2.py](../../model_v2.py))
- Config: `n_layer={cfg.n_layer}`, `n_head={cfg.n_head}`, `n_embd={cfg.n_embd}`, `block_size={cfg.block_size}`, `vocab_size={cfg.vocab_size}`
- **Parameters:** {n_params:,}  ({n_params/1e6:.2f} M)
- Modern components: RoPE, RMSNorm, ReLU² MLP, QK-Norm, tied embeddings, bias-free linears

## Training data (v3 mix — FineWeb-heavy)
- Mix: {", ".join(f"{k}={v:.0%}" for k, v in mix.items())}
- Loader: `MixedDataLoader` ([mix_loader.py](../../mix_loader.py))

## Optimizer (Phase 2)
- **Muon** for {n_muon_params:,} 2-D hidden parameters
  - lr={args.muon_lr}, momentum={args.muon_momentum}, ns_steps=5, nesterov=True, wd={args.weight_decay}
- **AdamW** for {n_adam_params:,} embedding + 1-D parameters
  - lr={args.adam_lr}, betas=(0.9, 0.95), wd_embed=0, wd_norm=0
- Both schedulers: warmup {args.warmup_steps} → cosine to {args.min_lr_ratio}× peak

## Hyperparameters
- micro_batch={args.micro_batch}, seq_len={args.seq_len}, total_batch_size={args.total_batch_size}
- max_steps={args.max_steps} (≈ {args.max_steps * args.total_batch_size / 1e9:.2f} B tokens)
- ckpt_every={args.ckpt_every}, eval_every={args.eval_every}
- grad_clip={args.grad_clip}

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
- Run script: `train_v3.py --run_name {args.run_name} --max_steps {args.max_steps}`

Checkpoints in this directory: every {args.ckpt_every} steps + final.
"""
    with open(os.path.join(out_dir, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
        f.write(card)


# ---------------------------------------------------------------------------
# Train

def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"v3_muon_{args.max_steps}st_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

    out_dir = os.path.join(args.log_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # DDP setup
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

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    if master:
        print(f"=== run_name: {args.run_name} ===")
        print(f"out_dir: {out_dir}")
        print(f"device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Batch math
    B, T = args.micro_batch, args.seq_len
    assert args.total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = args.total_batch_size // (B * T * ddp_world_size)
    if master:
        print(f"total_batch_size = {args.total_batch_size}")
        print(f"grad_accum_steps = {grad_accum_steps}")

    # Data — v3 mix
    train_loader = MixedDataLoader(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
        split="train", master_process=master, mix=V3_MIX,
    )
    val_loader = MixedDataLoader(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
        split="val", master_process=master, mix=V3_MIX,
    )

    torch.set_float32_matmul_precision("high")

    # Model — same arch as v2
    cfg = GPTConfigV2()
    raw_model = GPT(cfg).to(device)   # unwrapped GPT instance — used for ckpt save/load
    n_params = num_params(raw_model)
    if master:
        print(f"params: {n_params:,}  ({n_params/1e6:.2f} M)")
        assert n_params < 100_000_000, "OVER 100M PARAM CAP"

    # ----- Optimizer split: Muon hidden + AdamW embed/norms -----
    muon_params, adam_decay, adam_nodecay = split_params_for_muon(raw_model)
    n_muon = sum(p.numel() for p in muon_params)
    n_adam = sum(p.numel() for p in adam_decay) + sum(p.numel() for p in adam_nodecay)
    if master:
        print(f"Muon  → {len(muon_params)} tensors, {n_muon:,} params")
        print(f"AdamW → {len(adam_decay)+len(adam_nodecay)} tensors, {n_adam:,} params")

    muon_optim = Muon(
        muon_params,
        lr=args.muon_lr, momentum=args.muon_momentum,
        nesterov=True, ns_steps=5, weight_decay=args.weight_decay,
    )
    adam_optim = torch.optim.AdamW(
        [
            {"params": adam_decay,   "weight_decay": 0.0},   # embedding (no WD)
            {"params": adam_nodecay, "weight_decay": 0.0},   # 1-d / norms
        ],
        lr=args.adam_lr, betas=(0.9, 0.95), eps=1e-8,
        fused=(device_type == "cuda"),
    )

    # ---- Optional resume ----
    start_step = 0
    if args.resume is not None:
        if master:
            print(f"resuming from {args.resume}")
        rckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(rckpt["model"])
        # Newer ckpts include optimizer/loader/RNG state; old (pre-resume-support)
        # ckpts only have model+config+step. We restore what's available.
        if "muon_optim" in rckpt:
            muon_optim.load_state_dict(rckpt["muon_optim"])
        elif master:
            print("  (no muon_optim in old ckpt — Muon momenta start fresh)")
        if "adam_optim" in rckpt:
            adam_optim.load_state_dict(rckpt["adam_optim"])
        elif master:
            print("  (no adam_optim in old ckpt — AdamW m/v start fresh)")
        if "train_loader" in rckpt:
            train_loader.load_state(rckpt["train_loader"])
        elif master:
            print("  (no train_loader state — restarting source walks from 0)")
        if "val_loader" in rckpt:
            val_loader.load_state(rckpt["val_loader"])
        # CUDA + Python RNG (need CPU ByteTensor; map_location may have moved them)
        def _to_cpu_byte(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().to(torch.uint8)
            return t
        if "cuda_rng" in rckpt and rckpt["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(_to_cpu_byte(rckpt["cuda_rng"]))
        if "torch_rng" in rckpt and rckpt["torch_rng"] is not None:
            torch.set_rng_state(_to_cpu_byte(rckpt["torch_rng"]))
        start_step = int(rckpt["step"]) + 1
        if master:
            print(f"  → continuing at step {start_step} (was at step {rckpt['step']})")
            print(f"  → val_loss at resume: {rckpt.get('val_loss', float('nan')):.4f}")

    # DDP wrap (if applicable). model is what we forward through; raw_model
    # exposes the GPT instance for save/load regardless of wrapping.
    model = raw_model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    muon_min_lr = args.muon_lr * args.min_lr_ratio
    adam_min_lr = args.adam_lr * args.min_lr_ratio

    def lr_at(step, peak, floor):
        if step < args.warmup_steps:
            return peak * (step + 1) / args.warmup_steps
        if step >= args.max_steps:
            return floor
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + coeff * (peak - floor)

    # Logging — append on resume so the per-step trace is contiguous
    log_path = os.path.join(out_dir, "train_log.txt")
    log_mode = "a" if args.resume else "w"
    with open(log_path, log_mode) as f:
        if args.resume:
            f.write(f"\n# resumed at {datetime.datetime.now().isoformat()} from step {start_step}\n")
        else:
            f.write(f"# {args.run_name}\n# started {datetime.datetime.now().isoformat()}\n")

    final_val_loss = float("inf")
    started_at = time.time()

    for step in range(start_step, args.max_steps):
        t0 = time.time()
        last_step = (step == args.max_steps - 1)

        # ---- val pass ----
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

        # ---- checkpoint (full resume state) ----
        if master and step > 0 and (step % args.ckpt_every == 0 or last_step):
            ckpt = {
                "model": raw_model.state_dict(),
                "muon_optim": muon_optim.state_dict(),
                "adam_optim": adam_optim.state_dict(),
                "train_loader": train_loader.state(),
                "val_loader": val_loader.state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "config": cfg,
                "step": step,
                "val_loss": final_val_loss,
                "n_params": n_params,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(out_dir, f"model_{step:06d}.pt"))

        # ---- train step ----
        model.train()
        muon_optim.zero_grad()
        adam_optim.zero_grad()
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

        muon_lr_now = lr_at(step, args.muon_lr, muon_min_lr)
        adam_lr_now = lr_at(step, args.adam_lr, adam_min_lr)
        for g in muon_optim.param_groups:
            g["lr"] = muon_lr_now
        for g in adam_optim.param_groups:
            g["lr"] = adam_lr_now

        muon_optim.step()
        adam_optim.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        toks = B * T * grad_accum_steps * ddp_world_size
        if master:
            print(
                f"step {step:5d} | train {loss_acc.item():.4f} | "
                f"muon {muon_lr_now:.3e} adam {adam_lr_now:.3e} | "
                f"norm {norm:.3f} | dt {dt*1000:.0f}ms | tok/s {toks/dt:,.0f}"
            )
            with open(log_path, "a") as f:
                f.write(f"{step} train {loss_acc.item():.6f}\n")

    ended_at = time.time()

    # ---- val.bin perplexity ----
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
            n_muon_params=n_muon,
            n_adam_params=n_adam,
        )
        print(f"\nwrote {os.path.join(out_dir, 'MODEL_CARD.md')}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
