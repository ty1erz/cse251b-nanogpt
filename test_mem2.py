import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, V = 16, 1024, 50304
device = 'cuda'

logits = torch.randn(B, T, V, device=device, dtype=torch.bfloat16, requires_grad=True)
y = torch.randint(0, V, (B, T), device=device)

# standard autocast
torch.cuda.reset_peak_memory_stats()
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    cap = 30.0
    softcapped = cap * torch.tanh(logits / cap)
    main = F.cross_entropy(softcapped.view(-1, V), y.view(-1))
    main.backward()
print(f"Standard autocast CE peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

logits.grad = None
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# pure bfloat16
with torch.autocast(device_type='cuda', enabled=False):
    cap = 30.0
    softcapped = cap * torch.tanh(logits / cap)
    main = F.cross_entropy(softcapped.view(-1, V), y.view(-1))
    main.backward()
print(f"Pure bf16 CE peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
