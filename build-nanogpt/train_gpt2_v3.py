import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x, seq_len):
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)[None, None, :, :]
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)[None, None, :, :]
        return (x * cos) + (self.rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd == config.n_head * config.head_dim
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.gqa_repeat = config.n_head // config.n_kv_head
        self.dropout = config.dropout
        self.use_rope = config.position_embedding == "rope"
        self.use_qk_norm = config.qk_norm
        self.use_gated_attention = config.gated_attention
        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")

        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.rope = RotaryEmbedding(config.head_dim, config.block_size) if self.use_rope else None
        self.q_norm = RMSNorm(config.head_dim) if self.use_qk_norm else None
        self.k_norm = RMSNorm(config.head_dim) if self.use_qk_norm else None
        self.gate_proj = nn.Linear(config.n_embd, config.n_head, bias=False) if self.use_gated_attention else None

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        residual_input = x
        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            q = self.rope.apply_rotary(q, seq_len)
            k = self.rope.apply_rotary(k, seq_len)

        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.gqa_repeat, dim=1)
            v = v.repeat_interleave(self.gqa_repeat, dim=1)

        if self.has_sdpa:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.dropout > 0 and self.training:
                att = F.dropout(att, p=self.dropout)
            attn_out = att @ v

        if self.use_gated_attention:
            gate = torch.sigmoid(self.gate_proj(residual_input)).transpose(1, 2).unsqueeze(-1)
            attn_out = attn_out * gate

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_head * self.head_dim)
        return self.c_proj(attn_out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mlp_type == "swiglu"
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.down_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 18
    n_embd: int = 640
    n_head: int = 10
    n_kv_head: int = 5
    head_dim: int = 64
    intermediate_size: int = 1664
    bias: bool = False
    dropout: float = 0.0
    norm_type: str = "rmsnorm"
    position_embedding: str = "rope"
    mlp_type: str = "swiglu"
    attention_type: str = "gqa"
    factorized_embedding: bool = True
    token_emb_dim: int = 384
    qk_norm: bool = True
    gated_attention: bool = True
    tie_embeddings: bool = True
    data_roots_train: tuple = ("prepared_mixture_gpt2_full",)
    exp_name: str = "alvin_v3"
    optimizer: str = "muon_adamw"
    adamw_lr: float = 8e-5
    muon_lr: float = 2e-3
    min_adamw_lr: float = 8e-6
    min_muon_lr: float = 2e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_ratio: float = 0.01
    total_batch_size_tokens: int = 131072
    micro_batch_size: int = 64
    eval_interval: int = 500
    save_interval: int = 1000
    source_sampling_mode: str = "60_40"
    source_log_interval: int = 100
    split_idx: Optional[int] = None
    dataset_ranges: Optional[List[Dict]] = None
    val_bins: Optional[List[Dict]] = None


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        modules = {
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": RMSNorm(config.n_embd),
        }
        if config.factorized_embedding:
            modules["token_embedding"] = nn.Embedding(config.vocab_size, config.token_emb_dim)
            modules["input_proj"] = nn.Linear(config.token_emb_dim, config.n_embd, bias=False)
            modules["output_proj"] = nn.Linear(config.n_embd, config.token_emb_dim, bias=False)
        else:
            modules["wte"] = nn.Embedding(config.vocab_size, config.n_embd)
        if config.position_embedding == "learned":
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        if not config.factorized_embedding:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            if config.tie_embeddings:
                self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        bsz, seq_len = idx.size()
        assert seq_len <= self.config.block_size
        if self.config.factorized_embedding:
            x = self.transformer.input_proj(self.transformer.token_embedding(idx))
        else:
            x = self.transformer.wte(idx)
        if self.config.position_embedding == "learned":
            pos = torch.arange(0, seq_len, device=idx.device, dtype=torch.long)
            x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if self.config.factorized_embedding:
            h = self.transformer.output_proj(x)
            logits = h @ self.transformer.token_embedding.weight.t()
        else:
            logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def get_num_params(self):
        params = [p for p in self.parameters() if p.requires_grad]
        seen = set()
        total = 0
        for p in params:
            ptr = p.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            total += p.numel()
        return total

    def configure_optimizers(self, config: GPTConfig, device_type: str, master_process: bool):
        return build_optimizer(self, config, device_type, master_process)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.1, ns_steps=5, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize(update, steps, eps):
        orig_dtype = update.dtype
        x = update.float()
        transpose = False
        if x.size(0) > x.size(1):
            x = x.t()
            transpose = True
        x = x / (x.norm() + eps)
        eye = torch.eye(x.size(0), device=x.device, dtype=x.dtype)
        for _ in range(steps):
            a = x @ x.t()
            x = (1.5 * x) - (0.5 * (a @ x))
            x = 0.5 * x + 0.5 * (eye @ x)
        if transpose:
            x = x.t()
        return x.to(orig_dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = buf
                if update.ndim != 2:
                    p.add_(update, alpha=-lr)
                    continue
                update = self._orthogonalize(update, ns_steps, eps)
                p.add_(update, alpha=-lr)
        return loss


class HybridOptimizer:
    def __init__(self, adamw_opt, muon_opt=None):
        self.adamw_opt = adamw_opt
        self.muon_opt = muon_opt

    def zero_grad(self, set_to_none=True):
        self.adamw_opt.zero_grad(set_to_none=set_to_none)
        if self.muon_opt is not None:
            self.muon_opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.muon_opt is not None:
            self.muon_opt.step()
        self.adamw_opt.step()

    def set_lrs(self, adamw_lr, muon_lr=None):
        for group in self.adamw_opt.param_groups:
            group["lr"] = adamw_lr
        if self.muon_opt is not None and muon_lr is not None:
            for group in self.muon_opt.param_groups:
                group["lr"] = muon_lr

    def state_dict(self):
        state = {"adamw": self.adamw_opt.state_dict()}
        if self.muon_opt is not None:
            state["muon"] = self.muon_opt.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.adamw_opt.load_state_dict(state_dict["adamw"])
        if self.muon_opt is not None and "muon" in state_dict:
            self.muon_opt.load_state_dict(state_dict["muon"])


def build_optimizer(model: GPT, config: GPTConfig, device_type: str, master_process: bool):
    param_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
    adamw_names = []
    muon_names = []
    adamw_decay = []
    adamw_nodecay = []
    muon_params = []

    def use_muon(name, param):
        if config.optimizer != "muon_adamw":
            return False
        if param.ndim != 2:
            return False
        if "token_embedding" in name or "input_proj" in name or "output_proj" in name:
            return False
        if name.endswith("gate_proj.weight") and ".attn." in name:
            return False
        if "ln_" in name or "ln_f" in name or "q_norm" in name or "k_norm" in name:
            return False
        return True

    for name, param in param_dict.items():
        if use_muon(name, param):
            muon_names.append(name)
            muon_params.append(param)
        else:
            adamw_names.append(name)
            if param.ndim >= 2:
                adamw_decay.append(param)
            else:
                adamw_nodecay.append(param)

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    adamw_groups = [
        {"params": adamw_decay, "weight_decay": config.weight_decay},
        {"params": adamw_nodecay, "weight_decay": 0.0},
    ]
    adamw_opt = torch.optim.AdamW(
        adamw_groups,
        lr=config.adamw_lr,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
        fused=use_fused,
    )
    muon_opt = None
    if muon_params:
        muon_opt = Muon(muon_params, lr=config.muon_lr, momentum=config.beta2, weight_decay=config.weight_decay)

    if master_process:
        print(f"using fused AdamW: {use_fused}")
        print(f"AdamW params ({len(adamw_names)} tensors, {sum(param_dict[n].numel() for n in adamw_names):,} params):")
        for name in adamw_names:
            print(f"  adamw {name} {param_dict[name].numel():,}")
        print(f"Muon params ({len(muon_names)} tensors, {sum(param_dict[n].numel() for n in muon_names):,} params):")
        for name in muon_names:
            print(f"  muon  {name} {param_dict[name].numel():,}")

    return HybridOptimizer(adamw_opt, muon_opt)


def load_tokens(path):
    ext = os.path.splitext(path)[1]
    if ext == ".bin":
        return np.memmap(path, dtype=np.uint16, mode="r")
    return np.load(path, mmap_mode="r")


def infer_split_idx(data_root: str, fallback_total_tokens: int) -> int:
    meta_path = os.path.join(data_root, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        datasets = meta.get("datasets", [])
        if datasets:
            return int(datasets[0].get("train_tokens", fallback_total_tokens))
    raise RuntimeError(f"Could not infer split_idx from {meta_path}")


def build_dataset_ranges(config: GPTConfig, total_tokens: int) -> Optional[List[Dict]]:
    if config.dataset_ranges is not None:
        return config.dataset_ranges
    mode = config.source_sampling_mode
    if mode == "random":
        return None
    split_idx = config.split_idx
    train_root = config.data_roots_train[0] if config.data_roots_train else None
    if split_idx is None:
        if train_root is None or not os.path.isdir(train_root):
            raise RuntimeError("split_idx must be provided when dataset_ranges are enabled")
        split_idx = infer_split_idx(train_root, total_tokens)
    weights = {
        "100_fineweb": (1.0, 0.0),
        "90_10": (0.9, 0.1),
        "80_20": (0.8, 0.2),
        "60_40": (0.6, 0.4),
    }
    if mode not in weights:
        raise ValueError(f"Unknown source_sampling_mode: {mode}")
    fineweb_w, cosmo_w = weights[mode]
    return [
        {"name": "fineweb", "start": 0, "end": int(split_idx), "weight": fineweb_w},
        {"name": "cosmopedia", "start": int(split_idx), "end": int(total_tokens), "weight": cosmo_w},
    ]


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_roots: Optional[Sequence[str]] = None, dataset_ranges: Optional[List[Dict]] = None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.dataset_ranges = dataset_ranges if split == "train" else None
        self.source_counts = {}
        if data_roots is None:
            data_roots = ("prepared_mixture_gpt2_full",)
        shards = []
        for root in data_roots:
            if os.path.isdir(root):
                for fname in sorted(os.listdir(root)):
                    full = os.path.join(root, fname)
                    if split in fname and (fname.endswith(".bin") or fname.endswith(".npy")):
                        shards.append(full)
                if split == "train":
                    train_bin = os.path.join(root, "train.bin")
                    if os.path.exists(train_bin):
                        shards.append(train_bin)
            elif os.path.isfile(root):
                shards.append(root)
        deduped = []
        seen = set()
        for shard in shards:
            if shard not in seen:
                deduped.append(shard)
                seen.add(shard)
        self.shards = deduped
        assert len(self.shards) > 0, f"no shards found for split {split}"
        self.shard_token_counts = [len(load_tokens(s)) for s in self.shards]
        self.total_tokens = int(sum(self.shard_token_counts))
        self.rng = np.random.default_rng(1337 + process_rank)
        if self.dataset_ranges is not None:
            assert len(self.shards) == 1, "range-based sampling currently expects a single train.bin"
            self.tokens = load_tokens(self.shards[0])
            weights = np.asarray([max(0.0, float(r["weight"])) for r in self.dataset_ranges], dtype=np.float64)
            weights = weights / weights.sum()
            self.range_weights = weights
            self.source_counts = {r["name"]: 0 for r in self.dataset_ranges}
        elif split == "train":
            valid_lengths = [max(0, n - (self.T + 1)) for n in self.shard_token_counts]
            total_valid = sum(valid_lengths)
            self.train_shard_probs = np.asarray(valid_lengths, dtype=np.float64) / float(total_valid)
        else:
            self.reset()
        if master_process:
            print(f"found {len(self.shards)} shards for split {split}")
            if self.dataset_ranges is not None:
                print("dataset ranges:")
                for r in self.dataset_ranges:
                    print(f"  {r['name']}: start={r['start']:,} end={r['end']:,} weight={r['weight']:.3f}")

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def consume_source_counts(self):
        counts = dict(self.source_counts)
        for key in self.source_counts:
            self.source_counts[key] = 0
        return counts

    def _sample_from_ranges(self):
        x = torch.empty((self.B, self.T), dtype=torch.long)
        y = torch.empty((self.B, self.T), dtype=torch.long)
        choices = self.rng.choice(len(self.dataset_ranges), size=self.B, p=self.range_weights)
        for i, source_idx in enumerate(choices):
            source = self.dataset_ranges[int(source_idx)]
            low = int(source["start"])
            high = int(source["end"]) - (self.T + 1)
            if high < low:
                raise RuntimeError(f"dataset range {source['name']} is too short for T={self.T}")
            start = int(self.rng.integers(low, high + 1))
            buf = np.asarray(self.tokens[start : start + self.T + 1], dtype=np.int64)
            x[i] = torch.from_numpy(buf[:-1].copy())
            y[i] = torch.from_numpy(buf[1:].copy())
            self.source_counts[source["name"]] += 1
        return x, y

    def _sample_random(self):
        shard_idx = int(self.rng.choice(len(self.shards), p=self.train_shard_probs))
        tokens = load_tokens(self.shards[shard_idx])
        max_start = len(tokens) - (self.T + 1)
        starts = self.rng.integers(0, max_start + 1, size=self.B)
        x = torch.empty((self.B, self.T), dtype=torch.long)
        y = torch.empty((self.B, self.T), dtype=torch.long)
        for i, start in enumerate(starts):
            buf = np.asarray(tokens[start : start + self.T + 1], dtype=np.int64)
            x[i] = torch.from_numpy(buf[:-1].copy())
            y[i] = torch.from_numpy(buf[1:].copy())
        return x, y

    def next_batch(self):
        if self.split == "train":
            if self.dataset_ranges is not None:
                return self._sample_from_ranges()
            return self._sample_random()
        buf = self.tokens[self.current_position : self.current_position + self.B * self.T + 1]
        x = torch.tensor(np.asarray(buf[:-1], dtype=np.int64), dtype=torch.long).view(self.B, self.T)
        y = torch.tensor(np.asarray(buf[1:], dtype=np.int64), dtype=torch.long).view(self.B, self.T)
        self.current_position += self.B * self.T * self.num_processes
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


def build_default_val_bins():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    official = os.path.join(root, "val.bin")
    return [{"name": "official", "path": official}] if os.path.exists(official) else []


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
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size_tokens = 131072
micro_batch_size = 64
block_size = 1024
assert total_batch_size_tokens % (micro_batch_size * block_size * ddp_world_size) == 0
grad_accum_steps = total_batch_size_tokens // (micro_batch_size * block_size * ddp_world_size)
if master_process:
    print(f"total desired batch size tokens: {total_batch_size_tokens}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision("high")

model_config = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=18,
    n_embd=640,
    n_head=10,
    n_kv_head=5,
    head_dim=64,
    intermediate_size=1664,
    bias=False,
    dropout=0.0,
    norm_type="rmsnorm",
    position_embedding="rope",
    mlp_type="swiglu",
    attention_type="gqa",
    factorized_embedding=True,
    token_emb_dim=384,
    qk_norm=True,
    gated_attention=True,
    optimizer="muon_adamw",
    adamw_lr=8e-5,
    muon_lr=2e-3,
    min_adamw_lr=8e-6,
    min_muon_lr=2e-4,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    warmup_ratio=0.01,
    total_batch_size_tokens=65536,
    micro_batch_size=8,
    eval_interval=1000,
    save_interval=1000,
    data_roots_train=("prepared_mixture_gpt2_full",),
    source_sampling_mode="80_20",
    exp_name="alvin_v3",
    val_bins=build_default_val_bins(),
)

# range config from train.bin boundary
train_root = model_config.data_roots_train[0]
train_loader_probe = DataLoaderLite(B=1, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_roots=model_config.data_roots_train, dataset_ranges=None)
model_config.dataset_ranges = build_dataset_ranges(model_config, train_loader_probe.total_tokens)
train_loader = DataLoaderLite(
    B=model_config.micro_batch_size,
    T=model_config.block_size,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train",
    data_roots=model_config.data_roots_train,
    dataset_ranges=model_config.dataset_ranges,
)

model = GPT(model_config)
param_count = model.get_num_params()
if master_process:
    print(f"total parameters: {param_count:,}")
if param_count > 100_000_000:
    raise RuntimeError(f"V3 model has {param_count:,} parameters, exceeds 100M")
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_steps = math.ceil(train_loader.total_tokens / model_config.total_batch_size_tokens)
warmup_steps = max(1, int(max_steps * model_config.warmup_ratio))
if master_process:
    print(f"train tokens available: {train_loader.total_tokens:,}")
    print(f"max_steps for one pass over train.bin: {max_steps:,}")


def run_sanity_test(model_obj, device_obj):
    model_obj.eval()
    with torch.no_grad():
        idx = torch.randint(0, 50257, (2, 16), device=device_obj, dtype=torch.long)
        targets = torch.randint(0, 50257, (2, 16), device=device_obj, dtype=torch.long)
        logits, loss = model_obj(idx, targets)
        assert logits.shape == (2, 16, 50257)
        assert torch.isfinite(loss)
        print(f"sanity: logits={tuple(logits.shape)} loss={loss.item():.6f} params={model_obj.get_num_params():,}")
    model_obj.train()

if master_process:
    run_sanity_test(raw_model, device)


def get_lr(step):
    if step < warmup_steps:
        return model_config.adamw_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return model_config.min_adamw_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return model_config.min_adamw_lr + coeff * (model_config.adamw_lr - model_config.min_adamw_lr)


def get_muon_lr(step):
    if step < warmup_steps:
        return model_config.muon_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return model_config.min_muon_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return model_config.min_muon_lr + coeff * (model_config.muon_lr - model_config.min_muon_lr)


optimizer = raw_model.configure_optimizers(model_config, device_type, master_process)
amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device_type == "cuda" else torch.bfloat16)

log_dir = os.path.join("log", "v3")
os.makedirs(log_dir, exist_ok=True)
run_date = datetime.now().strftime("%Y%m%d")
log_file = os.path.join(log_dir, f"log_{run_date}_{model_config.exp_name}.txt")
with open(log_file, "w", encoding="utf-8") as f:
    pass

submission_template = os.path.join(os.path.dirname(__file__), "submission_model_template_v3.py")
submission_config_fields = tuple(asdict(model_config).keys())
evaluate_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluate.py"))


def export_model_bundle(model_obj, step, prefix):
    export_dir = os.path.join(log_dir, f"{prefix}_{run_date}_{model_config.exp_name}_{step:05d}")
    os.makedirs(export_dir, exist_ok=True)
    cfg = {k: getattr(model_obj.config, k) for k in submission_config_fields if hasattr(model_obj.config, k)}
    with open(os.path.join(export_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    shutil.copyfile(submission_template, os.path.join(export_dir, "model.py"))
    torch.save(model_obj.state_dict(), os.path.join(export_dir, "checkpoint.pt"))
    return export_dir


def run_eval_on_bin(model_dir, val_bin):
    output_json = os.path.join(model_dir, f"evaluate_{val_bin['name']}.json")
    cmd = [
        sys.executable,
        evaluate_py,
        "--model_dir", model_dir,
        "--data", val_bin["path"],
        "--device", device,
        "--output_json", output_json,
        "--expected_vocab_size", "50257",
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    with open(output_json, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results, elapsed


for step in range(max_steps):
    t0 = time.time()
    last_step = step == (max_steps - 1)
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = micro_step == (grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.grad_clip)
    adamw_lr = get_lr(step)
    muon_lr = get_muon_lr(step)
    optimizer.set_lrs(adamw_lr, muon_lr)
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    tokens_processed = model_config.micro_batch_size * model_config.block_size * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / max(dt, 1e-9)

    if master_process:
        print(
            f"step {step:5d} | train {loss_accum.item():.4f} | muon {muon_lr:.3e} | adam {adamw_lr:.3e} "
            f"| norm {float(grad_norm):.3f} | dt {dt*1000:.0f}ms | tok/s {tokens_per_sec:,.0f}"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
        if step > 0 and step % model_config.source_log_interval == 0:
            source_counts = train_loader.consume_source_counts()
            if source_counts:
                msg = " ".join(f"{k}={v}" for k, v in source_counts.items())
                print(f"sampled_windows {msg}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} sampled_windows {msg}\n")

        should_eval = ((step > 0 and step % model_config.eval_interval == 0) or last_step)
        should_save = ((step > 0 and step % model_config.save_interval == 0) or last_step)
        if should_eval:
            prefix = "submission" if should_save else "eval"
            export_dir = export_model_bundle(raw_model, step, prefix)
            if should_save:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} export {export_dir}\n")
            for val_bin in model_config.val_bins or []:
                results, eval_elapsed = run_eval_on_bin(export_dir, val_bin)
                ppl = results["perplexity"]
                avg_loss = results["avg_loss_nats"]
                total_eval_tokens = results["total_tokens_evaluated"]
                print(
                    f"evaluate {val_bin['name']} | ppl {ppl:.4f} | loss {avg_loss:.6f} "
                    f"| tok {total_eval_tokens:,} | sec {eval_elapsed:.2f}"
                )
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"{step} evaluate {val_bin['name']} ppl {ppl:.4f} loss {avg_loss:.6f} "
                        f"tok {total_eval_tokens} sec {eval_elapsed:.2f}\n"
                    )

if ddp:
    destroy_process_group()
