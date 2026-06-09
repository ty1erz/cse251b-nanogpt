"""
Model 3 low-learning-rate continuation (step-wise checkpoint search).

The depth-ablation main run (`train_model3_depth_ablation_one_epoch.py`) stops
at 38k updates. The report shows Model 3 is the strongest architecture at 38k
(20.918 public-validation PPL) but was never given the low-rate continuation
that Models 1 and 2 received. This script supplies exactly that continuation.

Method — a step-wise learning-rate staircase with checkpoint selection:

  * Each *stage* trains for a fixed number of updates at a CONSTANT learning
    rate (no within-stage decay).
  * During a stage, public `val.bin` perplexity is evaluated every
    `--eval_every` steps. The lowest-PPL checkpoint of the stage is mirrored to
    `<stage_dir>/best.pt`.
  * The next stage resumes from that `best.pt` at a LOWER constant LR.

This mirrors the report's "continuation as local search": every branch starts
from a strong saved state, lowers the schedule scale, and increases evaluation
frequency, while the architecture, data mixture, objective, optimizer split, and
loader position are all preserved. Selection is driven by the official `val.bin`
metric, NOT the internal mixed-shard validation loss.

Resume is seamless: model weights, Muon + AdamW optimizer state, both data
loader positions, and CPU/CUDA RNG state are restored from the checkpoint.

-----------------------------------------------------------------------------
Recommended staircase: 38k -> 43k in five 1,000-step stages. Each stage resumes
from the previous stage's best.pt and lowers the LR.

  # stage 1
  python model3_depth_ablation_one_epoch/train_model3_low_lr_finetune.py \
      --run_name model3_ft_s1 \
      --resume model3_depth_ablation_one_epoch/logs/model3_depth_20L_one_epoch/model_037999.pt \
      --max_steps 39000 --muon_lr 8e-4 --adam_lr 3.2e-5

  # stage 2
  python model3_depth_ablation_one_epoch/train_model3_low_lr_finetune.py \
      --run_name model3_ft_s2 \
      --resume model3_depth_ablation_one_epoch/logs/model3_ft_s1/best.pt \
      --max_steps 40000 --muon_lr 4e-4 --adam_lr 1.6e-5

  # stage 3
  python model3_depth_ablation_one_epoch/train_model3_low_lr_finetune.py \
      --run_name model3_ft_s3 \
      --resume model3_depth_ablation_one_epoch/logs/model3_ft_s2/best.pt \
      --max_steps 41000 --muon_lr 2e-4 --adam_lr 8e-6

  # stage 4
  python model3_depth_ablation_one_epoch/train_model3_low_lr_finetune.py \
      --run_name model3_ft_s4 \
      --resume model3_depth_ablation_one_epoch/logs/model3_ft_s3/best.pt \
      --max_steps 42000 --muon_lr 1e-4 --adam_lr 4e-6

  # stage 5 (final, smallest LR)
  python model3_depth_ablation_one_epoch/train_model3_low_lr_finetune.py \
      --run_name model3_ft_s5 \
      --resume model3_depth_ablation_one_epoch/logs/model3_ft_s4/best.pt \
      --max_steps 43000 --muon_lr 4e-5 --adam_lr 2.5e-6

The submission checkpoint is the lowest-PPL stage's best.pt.
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
EXPERIMENT_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(EXPERIMENT_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(EXPERIMENT_ROOT, "common"))
from model import GPT, GPTConfigV5, num_params  # noqa: E402
from mix_loader import MixedDataLoader  # noqa: E402
from muon import Muon, split_params_for_muon  # noqa: E402


# ---------------------------------------------------------------------------
# Config

V5_MIX = {
    "fineweb": 0.50,
    "wikipedia": 0.20,
    "science": 0.15,
    "books": 0.15,
}

DEFAULTS = dict(
    run_name="model3_ft_s1",
    log_root=os.path.join(HERE, "logs"),
    # resume is required — a continuation must start from a main-run checkpoint
    resume=None,
    # data
    total_batch_size=524288,
    micro_batch=16,
    seq_len=1024,
    # stage length — train up to max_steps at a constant LR, then pick best.pt
    max_steps=39000,
    ckpt_every=500,
    eval_every=250,
    # optim — CONSTANT LR for this stage (step-wise schedule). Lower it stage to
    # stage and resume from the previous stage's best.pt. No within-stage decay.
    muon_lr=8e-4,
    muon_momentum=0.95,
    adam_lr=3.2e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    # val.bin selection metric — lives at EXPERIMENT_ROOT/val.bin (override with VAL_BIN)
    val_bin_path=os.environ.get("VAL_BIN", os.path.join(EXPERIMENT_ROOT, "val.bin")),
    valbin_batch_size=8,
    valbin_max_batches=0,   # 0 = full val.bin; set e.g. 256 to speed up eval
    seed=1337,
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
# val.bin perplexity (matches evaluate.py; sliced to the 50257-class vocab)

@torch.no_grad()
def perplexity_on_valbin(model, val_path, device, block_size=1024,
                         batch_size=8, max_batches=0):
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
    n_done = 0
    for i in range(0, n_chunks, batch_size):
        x = torch.stack([data[j*block_size : j*block_size+block_size] for j in range(i, i+batch_size)]).to(device)
        y = torch.stack([data[j*block_size+1 : j*block_size+block_size+1] for j in range(i, i+batch_size)]).to(device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            logits, _ = model(x)
        logits = logits[:, :, :50257]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
        n_done += batch_size
        if max_batches and n_done >= max_batches * batch_size:
            break
    avg_loss = total_loss / total_tokens
    return dict(perplexity=math.exp(avg_loss), avg_loss_nats=avg_loss, tokens=total_tokens)


# ---------------------------------------------------------------------------
# Continuation stage

def main():
    args = parse_args()
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

    if args.resume is None or not os.path.exists(args.resume):
        raise FileNotFoundError(
            f"--resume checkpoint not found: {args.resume}\n"
            f"This is a continuation script; it must resume from a Model 3 checkpoint."
        )

    # Fail fast: val.bin drives model selection, so a missing file would waste
    # the whole stage (best.pt would never be written).
    if not os.path.exists(args.val_bin_path):
        raise FileNotFoundError(
            f"val.bin not found: {args.val_bin_path}\n"
            f"Set VAL_BIN=/path/to/val.bin or pass --val_bin_path explicitly."
        )

    if master:
        print(f"=== continuation run_name: {args.run_name} ===")
        print(f"out_dir: {out_dir}")
        print(f"device:  {device}")
        print(f"resume:  {args.resume}")

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

    # Data
    train_loader = MixedDataLoader(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
        split="train", master_process=master, mix=V5_MIX,
    )
    val_loader = MixedDataLoader(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
        split="val", master_process=master, mix=V5_MIX,
    )

    torch.set_float32_matmul_precision("high")

    # Load checkpoint first, rebuild the model from ITS config so the state dict
    # always matches (immune to any architecture-arg drift).
    rckpt = torch.load(args.resume, map_location=device, weights_only=False)
    src_cfg = rckpt["config"]
    cfg = GPTConfigV5(
        block_size=src_cfg.block_size,
        vocab_size=src_cfg.vocab_size,
        n_layer=src_cfg.n_layer,
        n_head=src_cfg.n_head,
        n_kv_head=getattr(src_cfg, "n_kv_head", src_cfg.n_head),
        n_embd=src_cfg.n_embd,
        rope_base=getattr(src_cfg, "rope_base", 10000.0),
        use_qk_norm=getattr(src_cfg, "use_qk_norm", True),
        logit_softcap=getattr(src_cfg, "logit_softcap", 0.0),
        mlp_hidden=getattr(src_cfg, "mlp_hidden", 1536),
    )
    raw_model = GPT(cfg).to(device)
    n_params = num_params(raw_model)
    if master:
        print(f"config: n_layer={cfg.n_layer} n_head={cfg.n_head} n_kv_head={cfg.n_kv_head} "
              f"n_embd={cfg.n_embd} mlp_hidden={cfg.mlp_hidden}")
        print(f"params: {n_params:,}  ({n_params/1e6:.2f} M)")
        assert n_params < 100_000_000, "OVER 100M PARAM CAP"

    # Optimizer split: Muon hidden + AdamW embed/norms
    muon_params, adam_decay, adam_nodecay = split_params_for_muon(raw_model)
    muon_optim = Muon(
        muon_params,
        lr=args.muon_lr, momentum=args.muon_momentum,
        nesterov=True, ns_steps=5, weight_decay=args.weight_decay,
    )
    adam_optim = torch.optim.AdamW(
        [
            {"params": adam_decay,   "weight_decay": 0.0},
            {"params": adam_nodecay, "weight_decay": 0.0},
        ],
        lr=args.adam_lr, betas=(0.9, 0.95), eps=1e-8,
        fused=(device_type == "cuda"),
    )

    # Restore full training state from the checkpoint.
    raw_model.load_state_dict(rckpt["model"])
    if "muon_optim" in rckpt:
        muon_optim.load_state_dict(rckpt["muon_optim"])
    elif master:
        print("  (no muon_optim in ckpt — Muon momenta start fresh)")
    if "adam_optim" in rckpt:
        adam_optim.load_state_dict(rckpt["adam_optim"])
    elif master:
        print("  (no adam_optim in ckpt — AdamW m/v start fresh)")
    if "train_loader" in rckpt:
        train_loader.load_state(rckpt["train_loader"])
    elif master:
        print("  (no train_loader state — restarting source walks from 0)")
    if "val_loader" in rckpt:
        val_loader.load_state(rckpt["val_loader"])

    def _to_cpu_byte(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().to(torch.uint8)
        return t
    if rckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(_to_cpu_byte(rckpt["cuda_rng"]))
    if rckpt.get("torch_rng") is not None:
        torch.set_rng_state(_to_cpu_byte(rckpt["torch_rng"]))

    start_step = int(rckpt["step"]) + 1
    if start_step >= args.max_steps:
        raise ValueError(
            f"Checkpoint is already at step {rckpt['step']} but --max_steps={args.max_steps}. "
            f"Increase --max_steps to continue."
        )
    if master:
        print(f"  -> resuming at step {start_step} (ckpt step {rckpt['step']})")
        print(f"  -> stage span: {start_step} -> {args.max_steps} ({args.max_steps - start_step} steps)")

    model = raw_model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Step-wise schedule: the LR is CONSTANT for this whole stage. To lower it,
    # start a new stage at smaller --muon_lr/--adam_lr, resuming from best.pt.

    log_path = os.path.join(out_dir, "finetune_log.txt")
    if master:
        with open(log_path, "a") as f:
            f.write(f"\n# stage started {datetime.datetime.now().isoformat()} "
                    f"from {args.resume} at step {start_step}\n")
            f.write(f"# constant lr: muon {args.muon_lr} adam {args.adam_lr} "
                    f"-> step {args.max_steps}\n")

    def save_ckpt(path, step, valbin):
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
            "val_loss": float("nan"),
            "valbin_perplexity": (valbin["perplexity"] if valbin else None),
            "n_params": n_params,
            "args": vars(args),
        }
        torch.save(ckpt, path)

    best_ppl = float("inf")
    best_step = -1
    started_at = time.time()

    for step in range(start_step, args.max_steps):
        t0 = time.time()
        last_step = (step == args.max_steps - 1)

        # val.bin eval (selection metric)
        valbin = None
        if step % args.eval_every == 0 or last_step:
            model.eval()
            if master:
                valbin = perplexity_on_valbin(
                    raw_model, args.val_bin_path, device,
                    block_size=args.seq_len,
                    batch_size=args.valbin_batch_size,
                    max_batches=args.valbin_max_batches,
                )
            if master and valbin is not None:
                ppl = valbin["perplexity"]
                flag = ""
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_step = step
                    save_ckpt(os.path.join(out_dir, "best.pt"), step, valbin)
                    flag = "  <- new best (saved best.pt)"
                print(f"step {step:5d} | val.bin ppl {ppl:.4f} | "
                      f"loss {valbin['avg_loss_nats']:.4f}{flag}")
                with open(log_path, "a") as f:
                    f.write(f"{step} valbin_ppl {ppl:.6f} valbin_loss {valbin['avg_loss_nats']:.6f}\n")

        # periodic stepped checkpoint
        if master and (step % args.ckpt_every == 0 or last_step) and step > start_step:
            save_ckpt(os.path.join(out_dir, f"model_{step:06d}.pt"), step, valbin)

        # train step
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

        muon_lr_now = args.muon_lr   # constant for the whole stage
        adam_lr_now = args.adam_lr
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

    # final full val.bin eval + summary
    if master:
        save_ckpt(os.path.join(out_dir, "last.pt"), args.max_steps - 1, None)
        print("\nfinal val.bin perplexity (full)...")
        final = perplexity_on_valbin(
            raw_model, args.val_bin_path, device,
            block_size=args.seq_len, batch_size=args.valbin_batch_size, max_batches=0,
        )
        if final is not None:
            print(f"final (step {args.max_steps - 1}) val.bin ppl: {final['perplexity']:.4f}")
            if final["perplexity"] < best_ppl:
                best_ppl = final["perplexity"]
                best_step = args.max_steps - 1
                save_ckpt(os.path.join(out_dir, "best.pt"), best_step, final)
                print("  <- final is the best; saved best.pt")
        elapsed_h = (time.time() - started_at) / 3600.0
        print(f"\n=== stage done in {elapsed_h:.2f} h ===")
        print(f"best val.bin ppl: {best_ppl:.4f} at step {best_step}  -> {os.path.join(out_dir, 'best.pt')}")
        with open(log_path, "a") as f:
            f.write(f"# BEST valbin_ppl {best_ppl:.6f} at step {best_step}\n")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
