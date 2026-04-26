"""
Local Learning Coefficient (LLC) estimator via Stochastic Gradient Langevin Dynamics.

Theory (Watanabe 2009):
  The free energy near a singular point w* satisfies:
      nF_n(w*) ≈ nL_n(w*) + λ log(n) - (m-1) log(log(n)) + O(1)

  where λ is the Real Log Canonical Threshold (RLCT), the Local Learning Coefficient.
  A lower λ indicates a more degenerate (singular) loss landscape — a simpler effective
  model. SLT explains why overparameterised networks generalise: they find singular optima
  with small λ, conferring an implicit complexity penalty.

Estimator (SGLD-based, following devinterp conventions):
  We sample from the tempered posterior p_β(w) ∝ exp(-βnL_n(w)) using SGLD at
  inverse temperature β = 1/log(n), then estimate:

      λ ≈ β · n · E_{w~p_β}[L_n(w) - L_n(w*)]

Localisation:
  A fixed step_size works well during early training but, once the model is in a sharp
  post-grokking minimum, the SGLD can escape the local basin and probe the global loss
  landscape rather than the local singular structure — biasing λ̂ upward.

  The remedy (following devinterp) is a *localisation* quadratic prior centred at w*:

      U_loc(w) = β·n·L_n(w)  +  (γ/2) · ‖w − w*‖²

  The extra spring force  γ·(w − w*)  keeps the chain near w*.
  This gives the *localised LLC* — the singularity depth of the local basin at w*.
  Set γ=0 to recover the original estimator.
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


@torch.no_grad()
def _full_loss(model: nn.Module, criterion: nn.Module,
               loader: DataLoader, device: str) -> float:
    """Mean cross-entropy loss over the full loader."""
    model.eval()
    total, count = 0.0, 0
    for a, b, c in loader:
        a, b, c = a.to(device), b.to(device), c.to(device)
        logits = model(a, b)
        total += criterion(logits, c).item() * len(a)
        count += len(a)
    return total / count


def estimate_llc(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    n_steps: int = 200,
    step_size: float = 3e-6,
    beta: Optional[float] = None,
    localization: float = 10_000.0,
    device: str = "cpu",
    burnin: int = 50,
) -> tuple[float, list[float]]:
    """
    Estimate the Local Learning Coefficient at the current weights via localised SGLD.

    Parameters
    ----------
    model         : network at its current weights w*
    criterion     : loss function (reduction='mean')
    train_loader  : dataloader for the training set
    n_steps       : SGLD steps to collect after burnin (more → lower variance, slower)
    step_size     : SGLD step size ε
    beta          : inverse temperature; defaults to 1/log(n_train) — the WBIC choice
    localization  : γ ≥ 0, spring constant keeping SGLD near w*.
                    γ=0 → global SGLD; γ>0 → localised (recommended for sharp minima).
                    Typical range 1–1000; tune so SGLD energy trace looks like a
                    stationary noisy signal rather than a drift.
    burnin        : initial SGLD steps discarded to allow mixing

    Returns
    -------
    llc     : scalar LLC estimate λ̂
    energies: per-step (L(w_t) − L(w*)) values for diagnostics
    """
    n = len(train_loader.dataset)
    if beta is None:
        beta = 1.0 / math.log(n)

    # Stable baseline at w*
    baseline = _full_loss(model, criterion, train_loader, device)

    # Keep a frozen copy of w* for the localisation spring
    sgld_model = copy.deepcopy(model).to(device)
    w_star = {name: p.data.clone() for name, p in sgld_model.named_parameters()}

    sgld_model.train()
    train_iter = _infinite(train_loader)
    energies: list[float] = []

    scale = beta * n          # gradient scale factor (per full dataset)

    for step in range(n_steps + burnin):
        a, b, c = next(train_iter)
        a, b, c = a.to(device), b.to(device), c.to(device)

        sgld_model.zero_grad()
        loss = criterion(sgld_model(a, b), c)
        loss.backward()

        # SGLD update: energy = β·n·L_n(w) + (γ/2)·‖w−w*‖²
        # ∇U ≈ (scale/|B|)·∇L_B + γ·(w−w*)
        batch_scale = scale / len(a)
        with torch.no_grad():
            for name, param in sgld_model.named_parameters():
                if param.grad is None:
                    continue
                spring = localization * (param.data - w_star[name])
                noise  = torch.randn_like(param) * math.sqrt(2.0 * step_size)
                param.data -= step_size * (batch_scale * param.grad + spring)
                param.data += noise

        if step >= burnin:
            with torch.no_grad():
                batch_loss = criterion(sgld_model(a, b), c).item()
            energies.append(batch_loss - baseline)

    # λ̂ = β·n·E[L(w) − L(w*)]
    llc = beta * n * float(np.mean(energies))
    return llc, energies


def _infinite(loader: DataLoader):
    """Cycle a DataLoader forever."""
    while True:
        yield from loader
