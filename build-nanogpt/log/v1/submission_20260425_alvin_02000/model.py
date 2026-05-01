import json
import os
import __main__
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x, T):
        cos = self.cos_cached[:T].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:T].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self._rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.gqa_repeat = config.n_head // config.n_kv_head
        self.use_rope = config.position_embedding == "rope"
        self.rope = RotaryEmbedding(self.head_dim, config.block_size) if self.use_rope else None

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = self.rope.apply_rotary(q, T)
            k = self.rope.apply_rotary(k, T)
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.gqa_repeat, dim=1)
            v = v.repeat_interleave(self.gqa_repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.mlp_type == "gelu":
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.mlp_type = "gelu"
        elif config.mlp_type == "swiglu":
            self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
            self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
            self.mlp_type = "swiglu"
        else:
            raise ValueError(f"Unknown mlp_type: {config.mlp_type}")

    def forward(self, x):
        if self.mlp_type == "gelu":
            return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.ln_1 = norm_cls(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = norm_cls(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 16
    n_head: int = 10
    n_kv_head: int = 5
    n_embd: int = 640
    dropout: float = 0.0
    bias: bool = False
    norm_type: str = "rmsnorm"
    position_embedding: str = "rope"
    mlp_type: str = "swiglu"
    intermediate_size: int = 1536
    tie_embeddings: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        modules = {
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": norm_cls(config.n_embd),
        }
        if config.position_embedding == "learned":
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx):
        _, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block size {self.config.block_size}")
        x = self.transformer.wte(idx)
        if self.config.position_embedding == "learned":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


def _load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return GPTConfig(**json.load(f))
    return GPTConfig()


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    setattr(__main__, "GPTConfig", GPTConfig)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = _load_config()
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
