"""
Final architecture for the CSE 251B NanoGPT competition.

  Differences from model_v2:
    * MLP swapped from ReLU² (2 matrices, 4x expansion) to SwiGLU
      (3 matrices, 8/3x expansion rounded to multiple of 64).
      Param count per MLP is nearly identical, but the gating mechanism
      is more expressive — typical ~1-3% PPL win at matched params.

  Same as model_v2:
    * RoPE positional encoding
    * RMSNorm everywhere
    * QK-Normalization for stability
    * Bias-free linears
    * Tied input/output embeddings
    * Optional Gemma-2 style logit softcap

  Default config: 13 layers · 10 heads · 640 embd · head_dim 64.
  ~96.6 M total parameters with SwiGLU at n_embd=640, hidden=1728.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config

@dataclass
class GPTConfigFinal:
    block_size: int = 1024
    vocab_size: int = 50304     # padded; eval expects 50257 (slice at submission)
    n_layer: int = 13           # one more than v2/v3 default (was 12)
    n_head: int = 10
    n_embd: int = 640           # head_dim = 640/10 = 64
    rope_base: float = 10000.0
    use_qk_norm: bool = True
    logit_softcap: float = 0.0
    # SwiGLU hidden = round_to_64(8/3 * n_embd). For n_embd=640 → 1728.
    mlp_hidden: int = 1728


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
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Attention with RoPE + QK-Norm

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfigFinal):
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
# SwiGLU MLP
# SwiGLU(x) = proj( silu(gate(x)) * up(x) )

class MLP(nn.Module):
    def __init__(self, config: GPTConfigFinal):
        super().__init__()
        hidden = config.mlp_hidden
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_up   = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_up(x))


# ---------------------------------------------------------------------------
# Block

class Block(nn.Module):
    def __init__(self, config: GPTConfigFinal):
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
    def __init__(self, config: GPTConfigFinal):
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
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"sequence length {T} > block_size {self.config.block_size}"
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if self.config.logit_softcap > 0:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def num_params(model: GPT) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


if __name__ == "__main__":
    cfg = GPTConfigFinal()
    m = GPT(cfg)
    n = num_params(m)
    print(f"GPTConfigFinal = {cfg}")
    print(f"params: {n:,}  ({n/1e6:.2f} M)")
    assert n < 100_000_000, "OVER 100M PARAM CAP"
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = m(x, x)
    print(f"in: {tuple(x.shape)}  out: {tuple(logits.shape)}  loss: {loss.item():.3f}")
    print("ok")


# ---------------------------------------------------------------------------
# Evaluation wrapper

class EvalGPT(nn.Module):
    """Adapter required by evaluate.py: model(input_ids) -> logits[:, :, :50257]."""

    def __init__(self, model: GPT):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits, _ = self.model(input_ids)
        return logits[:, :, :50257].contiguous()


def _config_from_checkpoint(checkpoint) -> GPTConfigFinal:
    raw_cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(raw_cfg, GPTConfigFinal):
        return raw_cfg
    if raw_cfg is not None and hasattr(raw_cfg, "__dict__"):
        allowed = GPTConfigFinal.__dataclass_fields__
        values = {k: v for k, v in vars(raw_cfg).items() if k in allowed}
        if values:
            return GPTConfigFinal(**values)

    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    return GPTConfigFinal(
        n_layer=int(args.get("n_layer", GPTConfigFinal.n_layer)),
        mlp_hidden=int(args.get("mlp_hidden", GPTConfigFinal.mlp_hidden)),
    )


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break
        else:
            state = checkpoint
    else:
        state = checkpoint

    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load a train_final_v1/model_final checkpoint for cse251b evaluate.py.

    The returned module emits logits over the exact GPT-2 vocabulary size
    expected by evaluate.py: (batch, seq_len, 50257). Internally the trained
    model uses a padded 50304-vocab output for kernel-friendly matmuls.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(checkpoint)
    model = GPT(cfg)
    model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
    model.to(device)
    model.eval()
    return EvalGPT(model).to(device).eval()
