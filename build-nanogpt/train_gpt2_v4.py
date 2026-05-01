import json
import os
import importlib.util
import math
import shutil
import subprocess
import sys
import time
import inspect
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x * rsqrt(mean(x^2) + eps) * weight
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x, T):
        # x: [B, n_head_or_n_kv_head, T, head_dim]
        cos = self.cos_cached[:T].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:T].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self._rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must divide n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head for GQA"
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head

        # Support standard MHA (n_kv_head == n_head) and grouped-query attention.
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.gqa_repeat = config.n_head // config.n_kv_head
        self.use_rope = config.position_embedding == "rope"
        self.rope = RotaryEmbedding(self.head_dim, config.block_size) if self.use_rope else None
        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B, n_head, T, hs)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, n_kv_head, T, hs)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, n_kv_head, T, hs)

        if self.use_rope:
            q = self.rope.apply_rotary(q, T)
            k = self.rope.apply_rotary(k, T)

        # GQA: repeat K/V heads to match Q heads.
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.gqa_repeat, dim=1)
            v = v.repeat_interleave(self.gqa_repeat, dim=1)

        if self.has_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=(self.dropout if self.training else 0.0),
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1), float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.dropout > 0 and self.training:
                att = F.dropout(att, p=self.dropout)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlp_type = config.mlp_type
        if self.mlp_type == "gelu":
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU(approximate='tanh')
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.c_proj.NANOGPT_SCALE_INIT = 1
        elif self.mlp_type == "swiglu":
            self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
            self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
            self.down_proj.NANOGPT_SCALE_INIT = 1
        else:
            raise ValueError(f"Unknown mlp_type: {self.mlp_type}")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.mlp_type == "gelu":
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
        else:
            x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        x = self.dropout(x)
        return x

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
    # Core model shape
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 16
    n_head: int = 10
    n_kv_head: int = 5
    n_embd: int = 640
    # Regularization & projections
    dropout: float = 0.0
    bias: bool = False
    # Modern decoder switches (ablation-friendly)
    norm_type: str = "rmsnorm"              # {"layernorm", "rmsnorm"}
    position_embedding: str = "rope"        # {"learned", "rope"}
    mlp_type: str = "swiglu"                # {"gelu", "swiglu"}
    intermediate_size: int = 1536
    tie_embeddings: bool = True
    # Data hooks: point to curated mixtures of pretokenized shards/files.
    data_roots_train: tuple = ("prepared_mixture_gpt2_full",)
    data_mixture_note: str = (
        "Use curated mixtures, e.g. textbook-quality, clean wiki/books, and target-domain corpora."
    )
    exp_name: str = "default"

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm

        modules = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = norm_cls(config.n_embd),
        )
        if config.position_embedding == "learned":
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_embeddings:
            # Tie input embedding and output projection weights.
            self.lm_head.weight = self.transformer.wte.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward token embeddings and optional learned absolute positions
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb
        if self.config.position_embedding == "learned":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = x + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_kv_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_kv_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_kv_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_kv_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # Keep architecture compatible with GPT-2 checkpoints.
        config_args['norm_type'] = "layernorm"
        config_args['position_embedding'] = "learned"
        config_args['mlp_type'] = "gelu"
        config_args['tie_embeddings'] = True
        config_args['bias'] = True
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = [
            'attn.q_proj.weight', 'attn.k_proj.weight', 'attn.v_proj.weight', 'attn.c_proj.weight',
            'mlp.c_fc.weight', 'mlp.c_proj.weight',
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # map GPT-2 c_attn -> q_proj/k_proj/v_proj
        c_attn_k = [k for k in sd_keys_hf if k.endswith("attn.c_attn.weight")]
        c_attn_b = [k for k in sd_keys_hf if k.endswith("attn.c_attn.bias")]
        for wk, bk in zip(c_attn_k, c_attn_b):
            prefix = wk[:-len("attn.c_attn.weight")]
            qk, kk, vk = prefix + "attn.q_proj.weight", prefix + "attn.k_proj.weight", prefix + "attn.v_proj.weight"
            qb, kb, vb = prefix + "attn.q_proj.bias", prefix + "attn.k_proj.bias", prefix + "attn.v_proj.bias"
            c_attn_w = sd_hf[wk].t().contiguous()
            q_w, k_w, v_w = c_attn_w.split(config.n_embd, dim=0)
            c_attn_bv = sd_hf[bk]
            q_b, k_b, v_b = c_attn_bv.split(config.n_embd, dim=0)
            with torch.no_grad():
                sd[qk].copy_(q_w); sd[kk].copy_(k_w); sd[vk].copy_(v_w)
                sd[qb].copy_(q_b); sd[kb].copy_(k_b); sd[vb].copy_(v_b)

        skip_hf = set(c_attn_k + c_attn_b)
        for k in sd_keys_hf:
            if k in skip_hf:
                continue
            if any(k.endswith(w) for w in transposed) and k in sd:
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                if k not in sd:
                    continue
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, beta1=0.9, beta2=0.95):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), eps=1e-8, fused=use_fused)
        return optimizer

    def get_num_params(self, non_embedding=False):
        """
        Return trainable parameter count.
        If embeddings are tied, only count shared storage once.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        seen = set()
        n_params = 0
        for p in params:
            ptr = p.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            n_params += p.numel()
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            if hasattr(self.transformer, "wpe"):
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

# -----------------------------------------------------------------------------
import numpy as np
try:
    import tiktoken
except Exception:
    tiktoken = None

def load_tokens(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".bin":
        return np.memmap(filename, dtype=np.uint16, mode="r")
    else:
        return np.load(filename, mmap_mode="r")

class DataLoaderLite:
    def __init__(
        self,
        B,
        T,
        process_rank,
        num_processes,
        split,
        data_roots: Optional[tuple] = None,
        dataset_ranges: Optional[list] = None,
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        self.split = split
        self.dataset_ranges = dataset_ranges if split == "train" else None
        self.train_rng = np.random.default_rng(1337 + process_rank)
        self.source_counts = {}

        # get shard filenames from one or more curated data roots/files
        if data_roots is None:
            data_roots = ("prepared_mixture_gpt2_full",)
        shards = []
        for root in data_roots:
            if os.path.isdir(root):
                root_files = sorted(os.listdir(root))
                root_files = [os.path.join(root, s) for s in root_files if split in s and (s.endswith(".npy") or s.endswith(".bin"))]
                shards.extend(root_files)
            elif os.path.isfile(root):
                base = os.path.basename(root)
                if split in base and (base.endswith(".npy") or base.endswith(".bin")):
                    shards.append(root)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.shard_token_counts = [len(load_tokens(s)) for s in self.shards]
        self.total_tokens = int(sum(self.shard_token_counts))
        if self.dataset_ranges is not None:
            assert len(self.shards) == 1, "range-based sampling expects a single train.bin file"
            self.tokens = load_tokens(self.shards[0])
            weights = np.asarray([float(r["weight"]) for r in self.dataset_ranges], dtype=np.float64)
            weights = weights / weights.sum()
            self.range_weights = weights
            self.source_counts = {r["name"]: 0 for r in self.dataset_ranges}
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        if self.dataset_ranges is not None:
            return
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def consume_source_counts(self):
        counts = dict(self.source_counts)
        for key in self.source_counts:
            self.source_counts[key] = 0
        return counts

    def next_batch(self):
        B, T = self.B, self.T
        if self.dataset_ranges is not None:
            x = torch.empty((B, T), dtype=torch.long)
            y = torch.empty((B, T), dtype=torch.long)
            choices = self.train_rng.choice(len(self.dataset_ranges), size=B, p=self.range_weights)
            for i, source_idx in enumerate(choices):
                source = self.dataset_ranges[int(source_idx)]
                low = int(source["start"])
                high = int(source["end"]) - (T + 1)
                if high < low:
                    raise RuntimeError(f"dataset range {source['name']} is too short for T={T}")
                start = int(self.train_rng.integers(low, high + 1))
                buf = self.tokens[start : start + T + 1]
                arr = np.asarray(buf, dtype=np.int64)
                x[i] = torch.from_numpy(arr[:-1].copy())
                y[i] = torch.from_numpy(arr[1:].copy())
                self.source_counts[source["name"]] += 1
            return x, y
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = torch.tensor(np.asarray(buf[:-1], dtype=np.int64), dtype=torch.long).view(B, T) # inputs
        y = torch.tensor(np.asarray(buf[1:], dtype=np.int64), dtype=torch.long).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

try:
    enc = tiktoken.get_encoding("gpt2") if tiktoken is not None else None
except Exception:
    enc = None

total_batch_size = 262144 # larger effective batch to reduce total optimizer steps
B = 8 # micro batch size
T = 1024 # sequence length aligned with evaluate.py context window
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')

# create model
resume_checkpoint = "/data/fengfei/cse251b-nanogpt/build-nanogpt/log/submission_20260425_alvin_12000/checkpoint.pt"
resume_step = 12000
used_tokens = resume_step * total_batch_size
fineweb_total_tokens = 9953989344
cosmopedia_total_tokens = 3624349131
remaining_fineweb_tokens = fineweb_total_tokens - used_tokens
remaining_total_tokens = remaining_fineweb_tokens + cosmopedia_total_tokens
dataset_ranges_v4 = [
    {
        "name": "fineweb_remaining",
        "start": used_tokens,
        "end": fineweb_total_tokens,
        "weight": 0.7,
    },
    {
        "name": "cosmopedia",
        "start": fineweb_total_tokens,
        "end": fineweb_total_tokens + cosmopedia_total_tokens,
        "weight": 0.3,
    },
]

model_config = GPTConfig(
    block_size=1024,
    vocab_size=50257,  # GPT-2 vocabulary for README-compatible evaluation
    n_layer=16,
    n_head=10,
    n_kv_head=5,
    n_embd=640,
    dropout=0.0,
    bias=False,
    norm_type="rmsnorm",
    position_embedding="rope",
    mlp_type="swiglu",
    intermediate_size=1536,
    tie_embeddings=True,
    data_roots_train=("prepared_mixture_gpt2_full",),
    exp_name="alvin_v4",
)
train_loader = DataLoaderLite(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train",
    data_roots=model_config.data_roots_train,
    dataset_ranges=dataset_ranges_v4,
)

model = GPT(model_config)
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
state_dict = torch.load(resume_checkpoint, map_location=device)
raw_state_dict = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict
model.load_state_dict(raw_state_dict, strict=True)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 1e-4
min_lr = 1e-5
full_run_max_steps = math.ceil(train_loader.total_tokens / total_batch_size)
max_steps = full_run_max_steps
remaining_steps = math.ceil(remaining_total_tokens / total_batch_size)
warmup_ratio = 0.005
warmup_steps = max(1, int(max_steps * warmup_ratio))

if master_process:
    n_trainable = raw_model.get_num_params()
    print(f"model trainable parameters (deduped/tied-aware): {n_trainable:,}")
    print(f"train tokens available: {train_loader.total_tokens:,}")
    print(f"resume checkpoint: {resume_checkpoint}")
    print(f"resume step: {resume_step:,}")
    print(f"used tokens before v4: {used_tokens:,}")
    print(f"remaining fineweb tokens: {remaining_fineweb_tokens:,}")
    print(f"cosmopedia tokens available: {cosmopedia_total_tokens:,}")
    print(f"remaining total tokens for v4: {remaining_total_tokens:,}")
    print(f"max_steps for one pass over full train.bin: {max_steps:,}")
    print(f"remaining v4 steps: {remaining_steps:,}")
    print("v4 dataset ranges:")
    for r in dataset_ranges_v4:
        print(f"  {r['name']}: start={r['start']:,} end={r['end']:,} weight={r['weight']:.2f}")
    if n_trainable >= 100_000_000:
        raise RuntimeError(f"Model has {n_trainable:,} trainable params, must stay under 100M.")

def run_sanity_test(model_obj, device_obj):
    model_obj.eval()
    with torch.no_grad():
        B_test, T_test = 2, 16
        idx = torch.randint(0, model_obj.config.vocab_size, (B_test, T_test), device=device_obj, dtype=torch.long)
        targets = torch.randint(0, model_obj.config.vocab_size, (B_test, T_test), device=device_obj, dtype=torch.long)
        logits, loss = model_obj(idx, targets)
        print(f"sanity: logits shape={tuple(logits.shape)}, loss={loss.item():.6f}, params={model_obj.get_num_params():,}")
    model_obj.train()

if master_process:
    run_sanity_test(raw_model, device)

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=max_lr,
    device_type=device_type,
    beta1=0.9,
    beta2=0.95,
)

if device_type == "cuda":
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    amp_dtype = torch.bfloat16

# create the log directory we will write checkpoints to and log to
log_dir = os.path.join("log", "v4")
os.makedirs(log_dir, exist_ok=True)
run_date = datetime.now().strftime("%Y%m%d")
log_file = os.path.join(log_dir, f"log_{run_date}_{model_config.exp_name}.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

submission_template = os.path.join(os.path.dirname(__file__), "submission_model_template.py")
submission_config_fields = (
    "block_size",
    "vocab_size",
    "n_layer",
    "n_head",
    "n_kv_head",
    "n_embd",
    "dropout",
    "bias",
    "norm_type",
    "position_embedding",
    "mlp_type",
    "intermediate_size",
    "tie_embeddings",
)


def export_submission_bundle(model_obj, step):
    export_dir = os.path.join(log_dir, f"submission_{run_date}_{model_config.exp_name}_{step:05d}")
    os.makedirs(export_dir, exist_ok=True)
    cfg = {k: getattr(model_obj.config, k) for k in submission_config_fields}
    with open(os.path.join(export_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    shutil.copyfile(submission_template, os.path.join(export_dir, "model.py"))
    torch.save(model_obj.state_dict(), os.path.join(export_dir, "checkpoint.pt"))
    return export_dir


evaluate_spec = importlib.util.spec_from_file_location(
    "official_evaluate", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluate.py"))
)
official_evaluate = importlib.util.module_from_spec(evaluate_spec)
evaluate_spec.loader.exec_module(official_evaluate)
evaluate_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluate.py"))
public_val_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "val.bin"))


def run_official_eval_via_model_dir(model_dir):
    output_json = os.path.join(model_dir, "evaluate_results.json")
    cmd = [
        sys.executable,
        evaluate_py,
        "--model_dir", model_dir,
        "--data", public_val_bin,
        "--device", device,
        "--output_json", output_json,
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    with open(output_json, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results, elapsed

for local_step in range(remaining_steps):
    step = resume_step + local_step
    t0 = time.time()
    last_step = (local_step == remaining_steps - 1)

    # once in a while evaluate hellaswag
    if False and (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        if enc is not None:
            tokens = enc.encode("Hello, I'm a language model,")
        else:
            tokens = [0, 1, 2, 3]
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens) if (enc is not None and raw_model.config.vocab_size >= 50257) else str(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
        if step > resume_step and step % 100 == 0:
            counts = train_loader.consume_source_counts()
            if counts:
                msg = " ".join(f"{k}={v}" for k, v in counts.items())
                print(f"sampled_windows {msg}")
                with open(log_file, "a") as f:
                    f.write(f"{step} sampled_windows {msg}\n")
        if step > 0 and (step % 1000 == 0 or last_step):
            export_dir = export_submission_bundle(raw_model, step)
            eval_results, eval_elapsed = run_official_eval_via_model_dir(export_dir)
            with open(log_file, "a") as f:
                f.write(f"{step} export {export_dir}\n")
                f.write(
                    f"{step} evaluate ppl {eval_results['perplexity']:.4f} "
                    f"loss {eval_results['avg_loss_nats']:.6f} "
                    f"tok {eval_results['total_tokens_evaluated']} "
                    f"sec {eval_elapsed:.2f}\n"
                )
            print(
                f"evaluate.py val.bin | ppl {eval_results['perplexity']:.4f} "
                f"| loss {eval_results['avg_loss_nats']:.6f} "
                f"| tokens {eval_results['total_tokens_evaluated']:,} "
                f"| time {eval_elapsed:.1f}s"
            )

if ddp:
    destroy_process_group()
