"""
CSE 251B submission (v3) — wraps a build-nanogpt v2/v3 trained GPT for evaluate.py.

Architecture lineage:
    Trained by train_v3.py using model_v2.GPT (Phase 1 architecture):
    RoPE + RMSNorm + ReLU² MLP + QK-Norm + tied embeddings + bias-free linears.
    91.19M parameters.

Optimizer used during training (does NOT affect eval but documenting for reference):
    Muon for 2-D hidden weights + AdamW for embeddings/norms (Phase 2).

Handles three mismatches with the evaluator:
  1. Checkpoint dict shape: {'model', 'config', 'step', 'val_loss', 'n_params', 'args'}
  2. Underlying GPT.forward returns (logits, loss); evaluator wants just logits.
  3. Trained vocab_size=50304 (padded); evaluator hard-checks logits.shape[-1]==50257.

Place a checkpoint here as `checkpoint.pt` and run:
    python evaluate.py --model_dir submission_v3/ --data val.bin
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Model definitions (copied from build-nanogpt/model_v2.py)

@dataclass
class GPTConfigV2:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 640
    rope_base: float = 10000.0
    use_qk_norm: bool = True
    logit_softcap: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_fp = x.float()
        rms = x_fp.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp * rms).to(x.dtype) * self.weight


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


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfigV2):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
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
        self.transformer.wte.weight = self.lm_head.weight

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, config.block_size, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# Evaluator-facing wrapper

class EvalGPT(nn.Module):
    """Adapts GPT to the evaluator's contract: forward(input_ids) -> logits[..., :50257]."""

    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, input_ids):
        logits, _ = self.gpt(input_ids)
        return logits[:, :, :50257]


# -----------------------------------------------------------------------------
# Required entrypoint

def load_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    # The checkpoint was pickled from train_v3.py, which imported GPTConfigV2 from
    # the `model_v2` module. Pickle stores the qualified name `model_v2.GPTConfigV2`,
    # so on load it tries to import a `model_v2` module — which the submission dir
    # doesn't have. Inject a fake module that exposes our local class. We also
    # cover __main__ in case a future checkpoint is pickled from a script-as-main.
    import sys
    import types

    if "model_v2" not in sys.modules:
        fake = types.ModuleType("model_v2")
        fake.GPTConfigV2 = GPTConfigV2
        sys.modules["model_v2"] = fake
    else:
        sys.modules["model_v2"].GPTConfigV2 = GPTConfigV2
    sys.modules["__main__"].GPTConfigV2 = GPTConfigV2

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    src_cfg = ckpt["config"]

    # Rebuild as our local GPTConfigV2 (immune to drift in pickled class).
    config = GPTConfigV2(
        block_size=src_cfg.block_size,
        vocab_size=src_cfg.vocab_size,
        n_layer=src_cfg.n_layer,
        n_head=src_cfg.n_head,
        n_embd=src_cfg.n_embd,
        rope_base=getattr(src_cfg, "rope_base", 10000.0),
        use_qk_norm=getattr(src_cfg, "use_qk_norm", True),
        logit_softcap=getattr(src_cfg, "logit_softcap", 0.0),
    )

    gpt = GPT(config)
    gpt.load_state_dict(ckpt["model"])
    model = EvalGPT(gpt).to(device)
    model.eval()
    return model


if __name__ == "__main__":
    import os
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        print(f"no checkpoint at {ckpt_path} — copy one from build-nanogpt/log_v3/<run>/model_*.pt")
        raise SystemExit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_model(ckpt_path, device)
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}  ({n/1e6:.2f} M)")
    x = torch.randint(0, 50257, (2, 128), device=device)
    y = m(x)
    print(f"in:  {tuple(x.shape)}")
    print(f"out: {tuple(y.shape)}")
    assert y.shape == (2, 128, 50257)
    print("ok — ready for evaluate.py")
