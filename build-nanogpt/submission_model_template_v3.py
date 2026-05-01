import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.gqa_repeat = config.n_head // config.n_kv_head
        self.use_rope = config.position_embedding == "rope"
        self.use_qk_norm = config.qk_norm
        self.use_gated_attention = config.gated_attention

        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=config.bias)
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
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        if self.use_gated_attention:
            gate = torch.sigmoid(self.gate_proj(residual_input)).transpose(1, 2).unsqueeze(-1)
            attn_out = attn_out * gate
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_head * self.head_dim)
        return self.c_proj(attn_out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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


class GPT(nn.Module):
    def __init__(self, config):
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

    def forward(self, idx):
        _, seq_len = idx.size()
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
        return logits


def load_model(checkpoint_path: str, device: str) -> nn.Module:
    model_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json next to checkpoint: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    allowed = {field: cfg[field] for field in GPTConfig.__dataclass_fields__ if field in cfg}
    model = GPT(GPTConfig(**allowed))
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
