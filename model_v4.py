"""
Phase 1 architecture (v2) for the CSE 251B NanoGPT competition.

Differences from build-nanogpt's train_gpt2.py / train_gpt2_small.py:

  Architecture
  ------------
  - Drops learned positional embeddings (`wpe`); uses RoPE (rotary positional
    embeddings) inside attention instead.
  - LayerNorm -> RMSNorm everywhere.
  - GELU MLP -> ReLU² MLP (modded-nanogpt finding, ~1-2% PPL gain).
  - Adds QK-Normalization (RMSNorm on q and k) for training stability at scale.
  - All linears are bias-free (modern convention; tiny param + slight stability).
  - Tied input/output embeddings (kept from baseline).

  Capacity
  --------
  - Default config: 12 layers · 10 heads · 640 embd · head_dim 64.
  - ~91.2M total parameters — under the 100M contest cap with ~8M headroom.

  Vocab padding
  -------------
  - vocab_size defaults to 50304 (multiple of 128) for kernel-friendly matmul,
    even though the GPT-2 BPE only uses 50257. The submission wrapper slices
    logits to [:, :, :50257] before returning.

The forward signature returns (logits, loss) — same as train_gpt2_small. The
EvalGPT adapter at submission time slices logits and unpacks the tuple.
"""

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
    n_layer: int = 12           # train_v4 overrides to 13 → ~96.1M params
    n_head: int = 10
    n_embd: int = 640           # head_dim = 640/10 = 64
    rope_base: float = 10000.0
    use_qk_norm: bool = True
    # Phase 3: Gemma-2 style logit softcap. 0 = disabled (v2/v3 default).
    # Set to e.g. 30.0 to apply `cap * tanh(logits/cap)` in forward.
    logit_softcap: float = 0.0


# ---------------------------------------------------------------------------
# RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # cast to float32 for the reduction, matches LLaMA / GPT-NeoX practice
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
    """Half-half RoPE (LLaMA / GPT-NeoX style).

    x:   (B, n_head, T, head_dim)
    cos: (T, head_dim/2)   sin: (T, head_dim/2)
    """
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

    def forward(self, idx, targets=None, mtp_weight=0.0, z_weight=0.0):
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
        loss_parts = None  # (main_detached, mtp_detached, z_detached) when composite
        if targets is not None:
            V = logits.size(-1)
            # Disable autocast to prevent F.cross_entropy from casting logits to FP32.
            # Safe because softcap bounds logits to 30. Saves ~6.6 GB VRAM total!
            with torch.autocast(device_type=logits.device.type, enabled=False):
                main = F.cross_entropy(logits.view(-1, V), targets.view(-1))
                if mtp_weight > 0.0 or z_weight > 0.0:
                    # MTP +2: predict targets[:, 1:] from logits[:, :-1]
                    if mtp_weight > 0.0 and targets.size(1) > 1:
                        mtp = F.cross_entropy(
                            logits[:, :-1, :].reshape(-1, V),
                            targets[:, 1:].reshape(-1),
                        )
                    else:
                        mtp = logits.new_zeros(())
                    # z-loss in native dtype; safe because softcap bounds logits
                    log_z = torch.logsumexp(logits, dim=-1)
                    z = (log_z * log_z).mean()
                    loss = main + mtp_weight * mtp + z_weight * z
                    loss_parts = (main.detach(), mtp.detach(), z.detach())
                else:
                    loss = main
        return logits, loss, loss_parts

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params  = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        groups = [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": nodecay_params,  "weight_decay": 0.0},
        ]
        if master:
            n_decay = sum(p.numel() for p in decay_params)
            n_nodecay = sum(p.numel() for p in nodecay_params)
            print(f"decayed tensors: {len(decay_params)} ({n_decay:,} params)")
            print(f"non-decayed tensors: {len(nodecay_params)} ({n_nodecay:,} params)")
        import inspect
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_ok and device_type == "cuda"
        return torch.optim.AdamW(
            groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )


# ---------------------------------------------------------------------------
# Param-count helper (handy for ablations / sanity-check)

def num_params(model: GPT) -> int:
    """Counts unique parameters (tied weights counted once)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


if __name__ == "__main__":
    cfg = GPTConfigV2()
    m = GPT(cfg)
    n = num_params(m)
    print(f"GPTConfigV2 = {cfg}")
    print(f"params: {n:,}  ({n/1e6:.2f} M)")
    assert n < 100_000_000, "OVER 100M PARAM CAP"
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = m(x, x)
    print(f"in: {tuple(x.shape)}  out: {tuple(logits.shape)}  loss: {loss.item():.3f}")
    print("ok")
