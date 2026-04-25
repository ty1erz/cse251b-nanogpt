"""
CSE 251B submission — wraps a build-nanogpt trained GPT for evaluate.py.

Handles three mismatches with the evaluator:
  1. The saved checkpoint is a dict {'model', 'config', 'step', 'val_loss'}.
  2. The underlying GPT.forward returns (logits, loss), not just logits.
  3. Training used vocab_size=50304 (padded), but evaluator requires 50257.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# GPT model (copied from build-nanogpt/train_gpt2_small.py)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# Evaluator-facing wrapper

class EvalGPT(nn.Module):
    """Adapts GPT to the evaluator's contract: forward(input_ids) -> logits (..., 50257)."""

    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, input_ids):
        logits, _ = self.gpt(input_ids)
        return logits[:, :, :50257]


# -----------------------------------------------------------------------------
# Required entrypoint

def load_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    # The checkpoint was pickled from train_gpt2_small.py running as __main__,
    # so its stored GPTConfig references __main__.GPTConfig. Make that resolvable.
    import sys
    sys.modules['__main__'].GPTConfig = GPTConfig

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    # Rebuild as our local GPTConfig in case the pickled class differs.
    config = GPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
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
