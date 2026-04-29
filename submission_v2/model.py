"""
CSE 251B submission_v2 — wraps a model_v2 (v2 architecture) checkpoint for evaluate.py.

Handles mismatches with the evaluator:
  1. The saved checkpoint is a dict {'model', 'config', 'step', 'val_loss', ...}.
  2. The underlying GPT.forward returns (logits, loss), not just logits.
  3. Training used vocab_size=50304 (padded), but evaluator requires 50257.

Architecture: RoPE, RMSNorm, ReLU² MLP, QK-Norm, bias-free linears, tied embeddings.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config

@dataclass
class GPTConfigV2:
    block_size: int = 1024
    vocab_size: int = 50304     # padded; eval expects 50257 (slice at submission)
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 640           # head_dim = 640/10 = 64
    rope_base: float = 10000.0
    use_qk_norm: bool = True


# ---------------------------------------------------------------------------
# RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_fp = x.float()
        rms = x_fp.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp * rms).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# RoPE

def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Returns cos, sin tables of shape (max_seq_len, head_dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-half RoPE (LLaMA / GPT-NeoX style)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Attention with RoPE + QK-Norm

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfigV2):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ---------------------------------------------------------------------------
# ReLU² MLP

class MLP(nn.Module):
    def __init__(self, config: GPTConfigV2):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).pow(2)
        return self.c_proj(x)


# ---------------------------------------------------------------------------
# Block

class Block(nn.Module):
    def __init__(self, config: GPTConfigV2):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# GPT

class GPT(nn.Module):
    def __init__(self, config: GPTConfigV2):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # tied

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, config.block_size, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"sequence length {T} > block_size {self.config.block_size}"
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Evaluator-facing wrapper

class EvalGPT(nn.Module):
    """Adapts GPT to the evaluator's contract: forward(input_ids) -> logits (..., 50257)."""

    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, input_ids):
        logits, _ = self.gpt(input_ids)
        return logits[:, :, :50257]


# ---------------------------------------------------------------------------
# Required entrypoint

def load_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    # The checkpoint was pickled from train_v2.py which imports from model_v2,
    # so the stored GPTConfigV2 references model_v2.GPTConfigV2.
    # Register a fake module so torch.load can resolve it.
    import sys
    import types
    fake_mod = types.ModuleType("model_v2")
    fake_mod.GPTConfigV2 = GPTConfigV2
    sys.modules["model_v2"] = fake_mod
    sys.modules['__main__'].GPTConfigV2 = GPTConfigV2

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    # Rebuild as our local GPTConfigV2 in case the pickled class differs.
    config = GPTConfigV2(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        rope_base=config.rope_base,
        use_qk_norm=config.use_qk_norm,
    )
    gpt = GPT(config)
    gpt.load_state_dict(ckpt["model"])
    model = EvalGPT(gpt).to(device)
    model.eval()
    return model


if __name__ == "__main__":
    import os
    ckpt = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_model(ckpt, device)
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}")
    x = torch.randint(0, 50257, (2, 128), device=device)
    y = m(x)
    print(f"in  : {tuple(x.shape)}")
    print(f"out : {tuple(y.shape)}")
    assert y.shape == (2, 128, 50257)
    print("ok")
