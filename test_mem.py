import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, V = 16, 1024, 50304
device = 'cuda'

print(f"Base alloc: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Simulate forward
logits = torch.randn(B, T, V, device=device, dtype=torch.bfloat16, requires_grad=True)
y = torch.randint(0, V, (B, T), device=device)

print(f"Logits alloc: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    cap = 30.0
    softcapped = cap * torch.tanh(logits / cap)
    
    main = F.cross_entropy(softcapped.view(-1, V), y.view(-1))
    mtp = F.cross_entropy(softcapped[:, :-1].reshape(-1, V), y[:, 1:].reshape(-1))
    
    log_z = torch.logsumexp(softcapped, dim=-1)
    z = (log_z * log_z).mean()
    
    total = main + mtp + z

print(f"Post forward alloc: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Peak forward alloc: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

del softcapped
del log_z

total.backward()

print(f"Post backward alloc: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Peak backward alloc: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
