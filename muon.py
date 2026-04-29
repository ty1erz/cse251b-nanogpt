"""
Muon optimizer (Keller Jordan).

For 2-D weight matrices in transformer hidden layers (attn projections, MLP
projections). NOT for embeddings, lm_head, or 1-D scalar/norm weights — those
should stay on AdamW.

Reference: https://github.com/KellerJordan/Muon

Update rule:
  1. Apply heavy momentum: buf = m * buf + grad   (with nesterov adjustment)
  2. Orthogonalize via 5th-order Newton-Schulz iteration in bf16:
       g_orth = NS(buf)                          # makes g_orth approx semi-unitary
  3. Scale to match Adam's effective step size:
       g_final = g_orth * sqrt(max(rows/cols, 1))
  4. Apply: p -= lr * g_final

Hyperparameter conventions vs AdamW:
  - Muon needs much higher lr (typical: 0.02) than Adam (3e-4 ~ 1e-3).
  - Muon needs higher momentum (0.95) than Adam beta1 (0.9).
"""

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """5th-order Newton-Schulz iteration to approximate the orthogonalization
    of a matrix G (i.e., compute G * (G^T G)^{-1/2}).

    Computes in bfloat16 for speed; coefficients are non-convergent but tuned
    for max slope at zero (per Keller Jordan), giving fast convergence in 5 steps.
    """
    assert G.ndim == 2, f"Newton-Schulz requires 2D, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T  # always shorter dim first
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2-D weight matrices.

    Args:
        params: iterable of 2-D parameters
        lr: learning rate (typical: 0.02)
        momentum: momentum factor (typical: 0.95)
        nesterov: whether to use Nesterov momentum
        ns_steps: Newton-Schulz iteration count (default 5)
        weight_decay: decoupled weight decay (applied like AdamW; default 0)
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError(
                        f"Muon only handles 2-D params; got shape {tuple(p.shape)}. "
                        f"Put 1-D / embedding params in AdamW."
                    )

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                update = g.add(buf, alpha=mom) if nesterov else buf

                # decoupled weight decay (AdamW-style)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # orthogonalize via Newton-Schulz
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # spectral scale: Adam-equivalent step magnitude
                rows, cols = p.shape
                update.mul_((max(rows, cols) / min(rows, cols)) ** 0.5)

                p.add_(update, alpha=-lr)

        return loss


def split_params_for_muon(model: torch.nn.Module):
    """Routes model parameters into (muon_params, adam_decay_params, adam_nodecay_params).

      muon_params           : 2-D hidden weights (attn / mlp projections)
      adam_decay_params     : 2-D embedding weight (vocab × n_embd)
      adam_nodecay_params   : 1-D scalars (RMSNorm weights, biases if any)

    Skips duplicates (handles tied embeddings: wte.weight === lm_head.weight).
    """
    muon, adam_decay, adam_nodecay = [], [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))

        if p.ndim < 2:
            adam_nodecay.append(p)
        elif name.endswith(".wte.weight") or name.endswith("lm_head.weight"):
            adam_decay.append(p)  # embeddings (also covers tied lm_head)
        else:
            muon.append(p)
    return muon, adam_decay, adam_nodecay
